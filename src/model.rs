// SPDX-License-Identifier: Apache-2.0
use std::collections::HashMap;
use std::ops::Range;

use serde_json::Value;

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Default)]
pub struct PredictOptions {
    /// Zero-based boosting iteration. LightGBM treats zero as the first one.
    pub start_iteration: usize,
    /// `None` or `Some(0)` uses all remaining iterations.
    pub num_iteration: Option<usize>,
    pub raw_score: bool,
    pub pred_leaf: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    /// One value for regression or binary classification; one per class for
    /// multiclass classification.
    pub scores: Vec<f64>,
    /// One leaf index per selected tree, in model order.
    pub leaves: Option<Vec<usize>>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Missing {
    None,
    Zero,
    NaN,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Predicate {
    Numeric {
        feature: usize,
        threshold_bits: u64,
        missing: Missing,
        default_left: bool,
    },
    Categorical {
        feature: usize,
        categories: Vec<i32>,
    },
}

impl Predicate {
    fn feature(&self) -> usize {
        match self {
            Self::Numeric { feature, .. } | Self::Categorical { feature, .. } => *feature,
        }
    }

    pub(crate) fn goes_left(&self, features: &[f64]) -> bool {
        let mut value = features[self.feature()];
        match self {
            Self::Numeric {
                threshold_bits,
                missing,
                default_left,
                ..
            } => {
                // Matches LightGBM's Tree::NumericalDecision: NaN becomes zero
                // unless this node explicitly uses NaN as its missing type.
                if value.is_nan() && *missing != Missing::NaN {
                    value = 0.0;
                }
                if (*missing == Missing::Zero && value == 0.0)
                    || (*missing == Missing::NaN && value.is_nan())
                {
                    *default_left
                } else {
                    value <= f64::from_bits(*threshold_bits)
                }
            }
            Self::Categorical { categories, .. } => {
                if value.is_nan() || !(-2_147_483_648.0..2_147_483_648.0).contains(&value) {
                    return false;
                }
                // The C API receives numeric category codes and truncates them
                // towards zero before testing the stored category bitset.
                let category = value as i32;
                category >= 0 && categories.binary_search(&category).is_ok()
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Node {
    Leaf {
        index: usize,
        value: f64,
    },
    Split {
        predicate: usize,
        left: Box<Node>,
        right: Box<Node>,
    },
}

impl Node {
    fn predict(&self, features: &[f64], predicates: &[Predicate]) -> (usize, f64) {
        match self {
            Self::Leaf { index, value } => (*index, *value),
            Self::Split {
                predicate,
                left,
                right,
            } => {
                if predicates[*predicate].goes_left(features) {
                    left.predict(features, predicates)
                } else {
                    right.predict(features, predicates)
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
enum Objective {
    Identity,
    Binary { sigmoid: f64 },
    Multiclass,
}

/// An immutable LightGBM tree ensemble compiled from `Booster.dump_model()`
/// or `LGBM_BoosterDumpModel` JSON.
#[derive(Clone, Debug)]
pub struct Model {
    pub(crate) predicates: Vec<Predicate>,
    pub(crate) trees: Vec<Node>,
    num_features: usize,
    trees_per_iteration: usize,
    average_output: bool,
    objective: Objective,
}

impl Model {
    pub fn from_dump_json(json: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(json)?;
        let root = value
            .as_object()
            .ok_or_else(|| invalid("root is not an object"))?;
        let tree_info = root
            .get("tree_info")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("missing tree_info array"))?;
        if tree_info.is_empty() {
            return Err(invalid("tree_info is empty"));
        }
        let trees_per_iteration = get_usize(&value, "num_tree_per_iteration")?;
        if trees_per_iteration == 0 || tree_info.len() % trees_per_iteration != 0 {
            return Err(invalid("invalid num_tree_per_iteration"));
        }
        let num_features = match root.get("max_feature_idx").and_then(Value::as_i64) {
            Some(n) if n >= 0 => usize::try_from(n)
                .ok()
                .and_then(|n| n.checked_add(1))
                .ok_or_else(|| invalid("max_feature_idx is too large"))?,
            Some(-1) => 0,
            _ => root
                .get("feature_names")
                .and_then(Value::as_array)
                .map(Vec::len)
                .ok_or_else(|| invalid("missing feature count"))?,
        };
        let num_class = get_usize(&value, "num_class")?;
        let objective = parse_objective(
            root.get("objective")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("missing objective"))?,
            num_class,
            trees_per_iteration,
        )?;
        let average_output = root
            .get("average_output")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let mut predicates = Vec::new();
        let mut predicate_ids = HashMap::new();
        let mut trees = Vec::with_capacity(tree_info.len());
        for (i, tree) in tree_info.iter().enumerate() {
            if tree.get("is_linear").and_then(Value::as_bool) == Some(true) {
                return Err(Error::Unsupported("linear trees".into()));
            }
            if let Some(tree_index) = tree.get("tree_index").and_then(Value::as_u64) {
                if usize::try_from(tree_index).ok() != Some(i) {
                    return Err(invalid("tree_info is not in tree_index order"));
                }
            }
            let structure = tree
                .get("tree_structure")
                .ok_or_else(|| invalid("tree lacks tree_structure"))?;
            trees.push(parse_node(
                structure,
                num_features,
                &mut predicates,
                &mut predicate_ids,
            )?);
        }
        Ok(Self {
            predicates,
            trees,
            num_features,
            trees_per_iteration,
            average_output,
            objective,
        })
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn num_predicates(&self) -> usize {
        self.predicates.len()
    }

    /// Stable, non-cryptographic fingerprint of the ordered predicate slots.
    /// Leaf values and tree shape may change without rebuilding the postings.
    pub fn predicate_fingerprint(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        fn feed(hash: &mut u64, bytes: &[u8]) {
            for byte in bytes {
                *hash ^= u64::from(*byte);
                *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        feed(&mut hash, &(self.predicates.len() as u64).to_le_bytes());
        for predicate in &self.predicates {
            match predicate {
                Predicate::Numeric {
                    feature,
                    threshold_bits,
                    missing,
                    default_left,
                } => {
                    feed(&mut hash, &[0]);
                    feed(&mut hash, &(*feature as u64).to_le_bytes());
                    feed(&mut hash, &threshold_bits.to_le_bytes());
                    feed(&mut hash, &[*missing as u8, u8::from(*default_left)]);
                }
                Predicate::Categorical {
                    feature,
                    categories,
                } => {
                    feed(&mut hash, &[1]);
                    feed(&mut hash, &(*feature as u64).to_le_bytes());
                    feed(&mut hash, &(categories.len() as u64).to_le_bytes());
                    for category in categories {
                        feed(&mut hash, &category.to_le_bytes());
                    }
                }
            }
        }
        hash
    }

    pub fn num_iterations(&self) -> usize {
        self.trees.len() / self.trees_per_iteration
    }

    pub(crate) fn tree_range(&self, options: PredictOptions) -> Range<usize> {
        let start = options.start_iteration.min(self.num_iterations());
        let remaining = self.num_iterations() - start;
        let count = match options.num_iteration {
            None | Some(0) => remaining,
            Some(n) => n.min(remaining),
        };
        start * self.trees_per_iteration..(start + count) * self.trees_per_iteration
    }

    /// Number of score values returned per row.
    pub fn output_count(&self) -> usize {
        self.trees_per_iteration
    }

    /// Number of leaf indices returned per row for a prediction window.
    pub fn selected_tree_count(&self, options: PredictOptions) -> usize {
        self.tree_range(options).len()
    }

    pub(crate) fn finish_scores(&self, scores: &mut [f64], tree_count: usize, raw: bool) {
        if raw {
            return;
        }
        if self.average_output && tree_count != 0 {
            let divisor = (tree_count / self.trees_per_iteration) as f64;
            for score in scores.iter_mut() {
                *score /= divisor;
            }
        }
        match self.objective {
            Objective::Identity => {}
            Objective::Binary { sigmoid } => {
                scores[0] = 1.0 / (1.0 + (-sigmoid * scores[0]).exp());
            }
            Objective::Multiclass => {
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for score in scores.iter_mut() {
                    *score = (*score - max).exp();
                    sum += *score;
                }
                for score in scores.iter_mut() {
                    *score /= sum;
                }
            }
        }
    }

    pub fn predict_scalar(&self, features: &[f64], options: PredictOptions) -> Result<Prediction> {
        if features.len() != self.num_features {
            return Err(Error::InvalidModel(format!(
                "expected {} features, got {}",
                self.num_features,
                features.len()
            )));
        }
        let range = self.tree_range(options);
        let mut scores = vec![0.0; self.output_count()];
        let mut leaves = options.pred_leaf.then(|| Vec::with_capacity(range.len()));
        for tree_idx in range.clone() {
            let (leaf, value) = self.trees[tree_idx].predict(features, &self.predicates);
            scores[tree_idx % self.trees_per_iteration] += value;
            if let Some(ref mut leaves) = leaves {
                leaves.push(leaf);
            }
        }
        self.finish_scores(&mut scores, range.len(), options.raw_score);
        Ok(Prediction { scores, leaves })
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidModel(message.into())
}

fn get_usize(value: &Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| invalid(format!("missing or invalid {field}")))
}

fn parse_objective(text: &str, num_class: usize, trees_per_iteration: usize) -> Result<Objective> {
    let mut parts = text.split_whitespace();
    let name = parts.next().unwrap_or("");
    let objective = match name {
        "binary" => {
            if num_class != 1 || trees_per_iteration != 1 {
                return Err(invalid("binary tree count does not match objective"));
            }
            let mut sigmoid: f64 = 1.0;
            for part in parts {
                if let Some(value) = part.strip_prefix("sigmoid:") {
                    sigmoid = value.parse().map_err(|_| invalid("invalid sigmoid"))?;
                }
            }
            if !sigmoid.is_finite() || sigmoid <= 0.0 {
                return Err(invalid("sigmoid must be positive and finite"));
            }
            Objective::Binary { sigmoid }
        }
        "multiclass" => {
            if num_class < 2 || trees_per_iteration != num_class {
                return Err(invalid("multiclass tree count does not match num_class"));
            }
            Objective::Multiclass
        }
        "regression" | "regression_l2" | "regression_l1" | "huber" | "fair" | "quantile"
        | "mape" | "lambdarank" | "rank_xendcg" | "custom" => {
            if trees_per_iteration != 1 {
                return Err(Error::Unsupported(format!(
                    "objective {name} with {trees_per_iteration} trees per iteration"
                )));
            }
            Objective::Identity
        }
        _ => return Err(Error::Unsupported(format!("objective {name}"))),
    };
    Ok(objective)
}

fn parse_node(
    node: &Value,
    num_features: usize,
    predicates: &mut Vec<Predicate>,
    predicate_ids: &mut HashMap<Predicate, usize>,
) -> Result<Node> {
    if node.get("leaf_index").is_some() {
        if node.get("leaf_coeff").is_some() || node.get("leaf_features").is_some() {
            return Err(Error::Unsupported("linear leaves".into()));
        }
        let index = get_usize(node, "leaf_index")?;
        let value = node
            .get("leaf_value")
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid("missing leaf_value"))?;
        if !value.is_finite() {
            return Err(invalid("non-finite leaf_value"));
        }
        return Ok(Node::Leaf { index, value });
    }
    let feature = get_usize(node, "split_feature")?;
    if feature >= num_features {
        return Err(invalid("split_feature exceeds model feature count"));
    }
    let predicate = match node.get("decision_type").and_then(Value::as_str) {
        Some("<=") => {
            let threshold = node
                .get("threshold")
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid("numeric split lacks threshold"))?;
            if !threshold.is_finite() {
                return Err(invalid("non-finite numeric threshold"));
            }
            let missing = match node.get("missing_type").and_then(Value::as_str) {
                Some("None") => Missing::None,
                Some("Zero") => Missing::Zero,
                Some("NaN") => Missing::NaN,
                _ => return Err(invalid("invalid missing_type")),
            };
            let default_left = node
                .get("default_left")
                .and_then(Value::as_bool)
                .ok_or_else(|| invalid("missing default_left"))?;
            Predicate::Numeric {
                feature,
                threshold_bits: threshold.to_bits(),
                missing,
                default_left,
            }
        }
        Some("==") => {
            let threshold = node
                .get("threshold")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("categorical split lacks threshold"))?;
            let mut categories = Vec::new();
            if !threshold.is_empty() {
                for category in threshold.split("||") {
                    let category: i32 = category
                        .parse()
                        .map_err(|_| invalid("invalid categorical threshold"))?;
                    if category < 0 {
                        return Err(invalid("negative categorical threshold"));
                    }
                    categories.push(category);
                }
            }
            categories.sort_unstable();
            categories.dedup();
            Predicate::Categorical {
                feature,
                categories,
            }
        }
        _ => return Err(invalid("unsupported decision_type")),
    };
    let predicate_id = if let Some(&id) = predicate_ids.get(&predicate) {
        id
    } else {
        let id = predicates.len();
        predicate_ids.insert(predicate.clone(), id);
        predicates.push(predicate);
        id
    };
    let left = node
        .get("left_child")
        .ok_or_else(|| invalid("split lacks left_child"))?;
    let right = node
        .get("right_child")
        .ok_or_else(|| invalid("split lacks right_child"))?;
    Ok(Node::Split {
        predicate: predicate_id,
        left: Box::new(parse_node(left, num_features, predicates, predicate_ids)?),
        right: Box::new(parse_node(right, num_features, predicates, predicate_ids)?),
    })
}
