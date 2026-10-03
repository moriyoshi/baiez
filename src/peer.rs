// SPDX-License-Identifier: Apache-2.0
//! Read a model and packed index through yesnod's plugin peer socket.

use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

use memmap2::Mmap;
use yesno_core::container::{codec, Container, ContainerKind};
use yesno_core::OrdSet;
use yesno_plugin::abi::Status;
use yesno_plugin::channel::{read_frame, recv_fd};
use yesno_plugin::ipc::{batched_lane_offset, Block, Frame, LaneKind, VERSION};

use crate::{Error, ModelBundle, PreparedIndex, Result};

struct Peer {
    socket: UnixStream,
    arena: Option<Mmap>,
    buffer: Vec<u8>,
    max_blocks: u32,
    max_lanes: usize,
}

fn peer_error(message: impl std::fmt::Display) -> Error {
    Error::Peer(message.to_string())
}

impl Peer {
    fn connect(path: &Path) -> Result<Self> {
        let socket = UnixStream::connect(path).map_err(peer_error)?;
        socket
            .set_read_timeout(Some(Duration::from_secs(30)))
            .map_err(peer_error)?;
        socket
            .set_write_timeout(Some(Duration::from_secs(30)))
            .map_err(peer_error)?;
        // Arena mode sends a one-byte protocol marker with an FD before the
        // greeting. Inline mode begins with the frame magic instead.
        let mut first = [0u8; 1];
        // SAFETY: `first` is a valid writable byte and `socket` is open. Peek
        // leaves the protocol byte/frame magic for the appropriate reader.
        let n = unsafe {
            libc::recv(
                socket.as_raw_fd(),
                first.as_mut_ptr().cast(),
                1,
                libc::MSG_PEEK,
            )
        };
        if n != 1 {
            return Err(peer_error(std::io::Error::last_os_error()));
        }
        let arena = if first[0] == VERSION {
            let (fd, version) = recv_fd(&socket).map_err(peer_error)?;
            if version != VERSION {
                return Err(peer_error(format!(
                    "peer protocol {version} is unsupported"
                )));
            }
            let file = std::fs::File::from(fd);
            // SAFETY: yesnod seals the memfd against shrink before handing it off.
            Some(unsafe { Mmap::map(&file) }.map_err(peer_error)?)
        } else {
            None
        };
        let mut peer = Self {
            socket,
            arena,
            buffer: Vec::new(),
            max_blocks: 1,
            max_lanes: 1,
        };
        match peer.recv()? {
            Frame::ServerHello {
                protocol,
                arena_bytes,
                max_blocks,
                max_lanes,
                ..
            } if protocol == VERSION as u32
                && max_blocks > 0
                && max_lanes > 0
                && peer.arena.as_ref().map_or(0, |map| map.len() as u64) == arena_bytes =>
            {
                peer.max_blocks = max_blocks;
                peer.max_lanes = max_lanes as usize;
            }
            other => return Err(peer_error(format!("invalid peer greeting: {other:?}"))),
        }
        peer.expect_done(Frame::ClientHello {
            protocol: VERSION as u32,
            name: "baiez".into(),
        })?;
        Ok(peer)
    }

    fn recv(&mut self) -> Result<Frame> {
        read_frame(&mut self.socket, &mut self.buffer)
            .map_err(peer_error)?
            .ok_or_else(|| peer_error("peer closed the socket"))
    }

    fn ask(&mut self, frame: Frame) -> Result<Frame> {
        self.socket
            .write_all(&frame.encode().map_err(peer_error)?)
            .map_err(peer_error)?;
        loop {
            match self.recv()? {
                Frame::Unavailable | Frame::Available { .. } | Frame::RoleChanged { .. } => {}
                Frame::GenerationChanged { .. } => {
                    return Err(Error::PeerStatus {
                        status: Status::GenerationChanged as u32,
                        message: "database generation changed during snapshot load".into(),
                    });
                }
                Frame::Fault { status, message } => {
                    return Err(Error::PeerStatus { status, message });
                }
                answer => return Ok(answer),
            }
        }
    }

    fn expect_done(&mut self, frame: Frame) -> Result<()> {
        match self.ask(frame)? {
            Frame::Done => Ok(()),
            answer => Err(peer_error(format!("unexpected peer response: {answer:?}"))),
        }
    }

    fn open_snapshot(&mut self) -> Result<u64> {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            match self.ask(Frame::SnapshotOpen) {
                Ok(Frame::SnapshotOpened { snapshot, .. }) => return Ok(snapshot),
                Err(Error::PeerStatus { status, .. })
                    if status == Status::Unavailable as u32 && Instant::now() < deadline =>
                {
                    std::thread::sleep(Duration::from_millis(250));
                }
                Err(error) => return Err(error),
                Ok(answer) => {
                    return Err(peer_error(format!(
                        "unexpected snapshot response: {answer:?}"
                    )));
                }
            }
        }
    }

    fn read_set(&mut self, snapshot: u64, key: u64) -> Result<OrdSet> {
        let (lanes, arena_off) = match self.ask(Frame::LanesAcquire {
            snapshot,
            keys: vec![key],
        })? {
            Frame::LanesAcquired { lanes, arena_off } => (lanes, arena_off),
            answer => return Err(peer_error(format!("unexpected lanes response: {answer:?}"))),
        };
        let result = self.read_lanes(lanes, arena_off);
        if result.is_ok() {
            self.expect_done(Frame::LanesRelease { lanes })?;
        }
        result
    }

    fn read_lanes(&mut self, lanes: u64, arena_off: u64) -> Result<OrdSet> {
        let mut chunks = Vec::new();
        let mut last_prefix = None;
        loop {
            match self.ask(Frame::BlockAdvanceMany {
                lanes,
                max_blocks: self.max_blocks,
            })? {
                Frame::BlockDone => break,
                Frame::Blocks { blocks } => {
                    if blocks.is_empty() {
                        break;
                    }
                    let arena = self
                        .arena
                        .as_ref()
                        .ok_or_else(|| peer_error("missing peer arena"))?;
                    for (block_index, block) in blocks.iter().enumerate() {
                        let lane = block
                            .lanes
                            .first()
                            .ok_or_else(|| peer_error("missing lane"))?;
                        if block.lanes.len() != 1 {
                            return Err(peer_error("unexpected peer lane count"));
                        }
                        let offset = usize::try_from(batched_lane_offset(
                            arena_off,
                            self.max_lanes,
                            block_index,
                            0,
                        ))
                        .map_err(peer_error)?;
                        let end = offset
                            .checked_add(lane.kind.payload_bytes(lane.count))
                            .ok_or_else(|| peer_error("lane offset overflow"))?;
                        let payload = arena
                            .get(offset..end)
                            .ok_or_else(|| peer_error("lane exceeds arena"))?;
                        push_block(&mut chunks, &mut last_prefix, block, payload)?;
                    }
                }
                Frame::BlocksInline { blocks, payload } => {
                    let mut offset = 0usize;
                    for block in &blocks {
                        let lane = block
                            .lanes
                            .first()
                            .ok_or_else(|| peer_error("missing lane"))?;
                        if block.lanes.len() != 1 {
                            return Err(peer_error("unexpected peer lane count"));
                        }
                        let end = offset
                            .checked_add(lane.kind.payload_bytes(lane.count))
                            .ok_or_else(|| peer_error("inline lane offset overflow"))?;
                        let bytes = payload
                            .get(offset..end)
                            .ok_or_else(|| peer_error("short inline lane"))?;
                        push_block(&mut chunks, &mut last_prefix, block, bytes)?;
                        offset = end;
                    }
                    if offset != payload.len() {
                        return Err(peer_error("trailing inline lane bytes"));
                    }
                    if blocks.is_empty() {
                        break;
                    }
                }
                answer => return Err(peer_error(format!("unexpected block response: {answer:?}"))),
            }
        }
        Ok(OrdSet::from_chunks(chunks))
    }
}

fn push_block(
    chunks: &mut Vec<(u64, Container)>,
    last_prefix: &mut Option<u64>,
    block: &Block,
    payload: &[u8],
) -> Result<()> {
    if last_prefix.is_some_and(|previous| block.prefix <= previous) {
        return Err(peer_error("peer blocks are not ascending"));
    }
    *last_prefix = Some(block.prefix);
    let lane = block.lanes[0];
    let decoded = match lane.kind {
        LaneKind::Absent => {
            if lane.count != 0 || !payload.is_empty() {
                return Err(peer_error("invalid absent lane"));
            }
            None
        }
        LaneKind::Array => Some(codec::decode(ContainerKind::Array, payload, lane.count)?),
        LaneKind::Bitmap => {
            let card = payload
                .chunks_exact(8)
                .map(|word| u64::from_le_bytes(word.try_into().unwrap()).count_ones())
                .sum();
            Some(codec::decode(ContainerKind::Bitmap, payload, card)?)
        }
        LaneKind::Run => {
            let run_count = u16::try_from(lane.count).map_err(peer_error)?;
            let mut encoded = Vec::with_capacity(payload.len() + 2);
            encoded.extend_from_slice(&run_count.to_le_bytes());
            for pair in payload.chunks_exact(4) {
                let start = u16::from_le_bytes([pair[0], pair[1]]);
                let end = u16::from_le_bytes([pair[2], pair[3]]);
                let length = end
                    .checked_sub(start)
                    .ok_or_else(|| peer_error("reversed run"))?;
                encoded.extend_from_slice(&start.to_le_bytes());
                encoded.extend_from_slice(&length.to_le_bytes());
            }
            Some(codec::decode(ContainerKind::Run, &encoded, 0)?)
        }
    };
    if let Some(container) = decoded {
        chunks.push((block.prefix, container));
    }
    Ok(())
}

/// Load an immutable model and prepared index from a local yesnod peer socket.
/// The metadata and packed view are read from the same remote snapshot. Scoring
/// uses owned yesno sets and requires no socket after this function returns.
pub fn load_peer_bundle(path: impl AsRef<Path>, key: u64) -> Result<(ModelBundle, PreparedIndex)> {
    let index_key = key
        .checked_add(1)
        .ok_or_else(|| Error::InvalidBundle("metadata key cannot be u64::MAX".into()))?;
    for attempt in 0..3 {
        let mut peer = Peer::connect(path.as_ref())?;
        let loaded = (|| {
            let snapshot = peer.open_snapshot()?;
            let metadata = peer.read_set(snapshot, key)?;
            let bundle = ModelBundle::from_set(&metadata, key)?;
            let descriptor = bundle
                .index()
                .ok_or_else(|| Error::InvalidBundle("model bundle has no packed index".into()))?;
            let packed = peer.read_set(snapshot, index_key)?;
            let prepared = descriptor.prepare_from_set(&packed, bundle.model())?;
            peer.expect_done(Frame::SnapshotClose { snapshot })?;
            Ok((bundle, prepared))
        })();
        match loaded {
            Err(Error::PeerStatus { status, .. })
                if status == Status::GenerationChanged as u32 && attempt < 2 =>
            {
                continue
            }
            result => return result,
        }
    }
    unreachable!()
}

/// Read only model metadata through a yesnod peer socket. This also works for
/// bundles that do not yet have a packed index.
pub fn load_peer_model(path: impl AsRef<Path>, key: u64) -> Result<ModelBundle> {
    for attempt in 0..3 {
        let mut peer = Peer::connect(path.as_ref())?;
        let loaded = (|| {
            let snapshot = peer.open_snapshot()?;
            let metadata = peer.read_set(snapshot, key)?;
            let bundle = ModelBundle::from_set(&metadata, key)?;
            peer.expect_done(Frame::SnapshotClose { snapshot })?;
            Ok(bundle)
        })();
        match loaded {
            Err(Error::PeerStatus { status, .. })
                if status == Status::GenerationChanged as u32 && attempt < 2 =>
            {
                continue
            }
            result => return result,
        }
    }
    unreachable!()
}
