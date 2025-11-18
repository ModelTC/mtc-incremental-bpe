use derive_more::Deref;
use smallvec::SmallVec;

use crate::{
    NormalizedDict, RuleId, TokenId,
    normalize::SINGLETON_PRIORITY,
    suf_suc::NUM_INLINE_FOREST_NODES,
    typed_vec::{TypedVec, typed_vec_index},
};

typed_vec_index!(pub(crate) ForestNodeId, u32);

pub(crate) const FOREST_VIRTUAL_ROOT: ForestNodeId = ForestNodeId::ZERO;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct SucNode {
    pub token_id: TokenId,
    pub priority: RuleId,
    pub skip_len: u32,
    pub pre_id: ForestNodeId,
    pub depth: u32,
    pub parent: ForestNodeId,
    pub subtree_last_node: ForestNodeId,
    pub children: SmallVec<[ForestNodeId; NUM_INLINE_FOREST_NODES]>,
}

#[derive(Debug, Deref)]
pub(crate) struct SucForest {
    #[deref]
    nodes: TypedVec<ForestNodeId, SucNode>,
    pub(crate) token_to_node_id: TypedVec<TokenId, ForestNodeId>,
}

impl SucForest {
    pub fn new(dict: &NormalizedDict) -> Self {
        let mut roots = Vec::with_capacity(NUM_INLINE_FOREST_NODES);
        let mut children: TypedVec<TokenId, _> = std::iter::repeat_n(
            SmallVec::<[TokenId; NUM_INLINE_FOREST_NODES]>::new(),
            dict.num_of_tokens().as_usize(),
        )
        .collect();
        for (token_id, rule_id) in dict.priorities.enumerate_copied() {
            if dict.is_single(token_id) {
                roots.push(token_id);
            } else if dict.is_useful(token_id) {
                children[dict[rule_id].suc].push(token_id);
            }
        }

        roots.sort();
        for vec in &mut children {
            vec.sort_by_key(|&i| !dict.priorities[i]);
        }

        let mut token_to_node_id: TypedVec<TokenId, ForestNodeId> =
            vec![FOREST_VIRTUAL_ROOT; dict.num_of_tokens().as_usize()].into();
        let virtual_root = SucNode {
            token_id: TokenId::MAX,
            priority: RuleId::MAX,
            skip_len: 0,
            depth: 0,
            pre_id: FOREST_VIRTUAL_ROOT,
            parent: FOREST_VIRTUAL_ROOT,
            subtree_last_node: FOREST_VIRTUAL_ROOT,
            children: Default::default(),
        };
        let mut nodes: TypedVec<ForestNodeId, SucNode> = vec![virtual_root].into();

        let mut alloc = {
            |token_id: TokenId, parent: ForestNodeId| {
                let node_id = nodes.len();
                nodes.push(SucNode {
                    token_id,
                    priority: dict.priorities[token_id],
                    skip_len: 1,
                    parent,
                    depth: 1,
                    pre_id: FOREST_VIRTUAL_ROOT,
                    subtree_last_node: node_id,
                    children: Default::default(),
                });
                node_id
            }
        };

        let mut stack = vec![(None::<TokenId>, 0usize)];

        while let Some((token_id, child_id)) = stack.last_mut() {
            if let Some(token_id) = *token_id {
                if *child_id >= children[token_id].len() {
                    stack.pop();
                    continue;
                }
                let child = children[token_id][*child_id];
                *child_id += 1;
                let parent = token_to_node_id[token_id];
                token_to_node_id[child] = alloc(child, parent);
                stack.push((Some(child), 0usize));
            } else {
                if *child_id >= roots.len() {
                    stack.pop();
                    continue;
                }
                let child = roots[*child_id];
                *child_id += 1;
                token_to_node_id[child] = alloc(child, FOREST_VIRTUAL_ROOT);
                stack.push((Some(child), 0usize));
            }
        }

        for node_id in nodes.keys().rev() {
            nodes[node_id].children.reverse();

            #[cfg(debug_assertions)]
            {
                for &child in &nodes[node_id].children {
                    debug_assert!(child > node_id && nodes[child].parent == node_id);
                    debug_assert!(
                        (node_id == FOREST_VIRTUAL_ROOT)
                            ^ (nodes[child].priority < SINGLETON_PRIORITY)
                    );
                }
            }

            if node_id == FOREST_VIRTUAL_ROOT {
                continue;
            }

            #[cfg(debug_assertions)]
            {
                debug_assert!(
                    nodes[node_id]
                        .children
                        .is_sorted_by_key(|&c| !nodes[c].priority)
                );
                for slice in nodes[node_id].children.windows(2) {
                    let u = slice[0];
                    let v = slice[1];
                    debug_assert!(u < v && nodes[u].priority > nodes[v].priority);
                }
            }

            let parent = nodes[node_id].parent;
            debug_assert!(parent < node_id);
            nodes[parent].subtree_last_node = nodes[parent]
                .subtree_last_node
                .max(nodes[node_id].subtree_last_node);
            nodes[parent].children.push(node_id);

            let rule_id = nodes[node_id].priority;
            if rule_id < SINGLETON_PRIORITY {
                debug_assert!(rule_id < dict.rules.len());
                let pre_id = token_to_node_id[dict[rule_id].pre];
                nodes[node_id].pre_id = pre_id;
            }
        }

        for node_id in nodes.keys() {
            if node_id == FOREST_VIRTUAL_ROOT {
                continue;
            }
            let parent_id = nodes[node_id].parent;
            nodes[node_id].depth = nodes[parent_id].depth + 1;
        }

        for token_id in {
            let mut order: Vec<_> = dict.tokens.keys().collect();
            order.sort_unstable_by_key(|&i| dict[i].len());
            order
        } {
            let node_id = token_to_node_id[token_id];
            if !dict.is_useful(token_id) {
                debug_assert!(node_id == FOREST_VIRTUAL_ROOT);
                continue;
            }
            debug_assert!(node_id != FOREST_VIRTUAL_ROOT);
            let node = &nodes[node_id];
            let (pre_id, parent) = (node.pre_id, node.parent);
            if node.parent == FOREST_VIRTUAL_ROOT {
                continue;
            }
            debug_assert!(node.pre_id != FOREST_VIRTUAL_ROOT);
            nodes[node_id].skip_len = nodes[pre_id].skip_len + nodes[parent].skip_len;
        }

        Self {
            token_to_node_id,
            nodes,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Dictionary, NormalizedDict, Vocab,
        successor::{FOREST_VIRTUAL_ROOT, SucForest},
    };

    #[test]
    fn test_suc_forest() {
        let vocab = Vocab::new([
            b"<unk>" as &[_],
            b"a",
            b"abc",
            b"abcde",
            b"abcdef",
            b"b",
            b"ba",
            b"bc",
            b"bcdef",
            b"c",
            b"cd",
            b"cde",
            b"cdefg",
            b"d",
            b"de",
            b"def",
            b"e",
            b"ef",
            b"efg",
            b"f",
            b"g",
        ])
        .unwrap();
        let dict = Dictionary::new_from_token_pair(
            vocab.clone(),
            [
                ("b", "c"),
                ("e", "f"),
                ("d", "e"),
                ("c", "d"),
                ("d", "ef"),
                ("b", "a"),
                ("a", "bc"),
                ("abc", "de"),
                ("abc", "def"),
                ("bc", "def"),
                ("c", "de"),
                ("ef", "g"),
                ("cd", "efg"),
            ],
        )
        .unwrap();
        let normalized = NormalizedDict::new_in_bytes(dict.clone());
        let forest = SucForest::new(&normalized);
        for (node_id, node) in forest.enumerate() {
            if node_id != FOREST_VIRTUAL_ROOT && node.token_id.inner() > 0 {
                assert!(vocab[node.token_id].len() == node.skip_len as usize);
            }
            let s = if node_id.0 == 0 {
                "(epsilon)"
            } else {
                std::str::from_utf8(&vocab[node.token_id]).unwrap()
            };
            println!("{s:12} {node_id:2}: {node:?}");
        }
        let normalized = NormalizedDict::new_in_utf8(dict.clone());
        let forest_b = SucForest::new(&normalized);
        assert!(
            forest
                .token_to_node_id
                .iter()
                .zip(forest_b.token_to_node_id.iter())
                .all(|(&i, &j)| i == j)
        );
        assert!(forest.iter().zip(forest_b.iter()).all(|(i, j)| i == j));
        let dict = Dictionary::new_from_token_pair(
            vocab.clone(),
            [
                ("b", "c"),
                ("e", "f"),
                ("d", "e"),
                ("c", "d"),
                ("d", "ef"),
                ("a", "bc"),
                ("b", "a"),
                ("abc", "de"),
                ("abc", "def"),
                ("bc", "def"),
                ("c", "de"),
                ("ef", "g"),
                ("cd", "efg"),
            ],
        )
        .unwrap();
        let normalized = NormalizedDict::new_in_bytes(dict.clone());
        let forest = SucForest::new(&normalized);
        for (node_id, node) in forest.enumerate() {
            if node_id != FOREST_VIRTUAL_ROOT && node.token_id.inner() > 0 {
                assert!(vocab[node.token_id].len() == node.skip_len as usize);
            }
            let s = if node_id.0 == 0 {
                "(epsilon)"
            } else {
                std::str::from_utf8(&vocab[node.token_id]).unwrap()
            };
            println!("{s:12} {node_id:2}: {node:?}");
        }
        let normalized = NormalizedDict::new_in_utf8(dict.clone());
        let forest_b = SucForest::new(&normalized);
        assert!(
            forest
                .token_to_node_id
                .iter()
                .zip(forest_b.token_to_node_id.iter())
                .all(|(&i, &j)| i == j)
        );
        assert!(forest.iter().zip(forest_b.iter()).all(|(i, j)| i == j));

        let dict = Dictionary::new_from_token_pair(
            vocab.clone(),
            [("b", "c"), ("e", "f"), ("abc", "def")],
        )
        .unwrap();
        let normalized = NormalizedDict::new_in_bytes(dict.clone());
        SucForest::new(&normalized);
    }
}
