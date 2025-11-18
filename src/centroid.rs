use std::collections::BTreeMap;

use derive_more::Deref;
use smallvec::SmallVec;

use crate::{
    successor::{FOREST_VIRTUAL_ROOT, ForestNodeId, SucForest, SucNode},
    suf_suc::{NUM_INLINE_FOREST_NODES, SufSucNode, SufSucNodeSet},
    typed_vec::{TypedVec, typed_vec_index},
};

typed_vec_index!(pub(crate) CentroidId, u16);

#[derive(Debug, Deref)]
pub(crate) struct CentroidNode {
    #[deref]
    node: SufSucNode,
    intervals: SmallVec<[(ForestNodeId, ForestNodeId); NUM_INLINE_FOREST_NODES]>,
    children: SmallVec<[CentroidId; NUM_INLINE_FOREST_NODES]>,
}

impl CentroidNode {
    fn new(node: SubTreeNodeRef) -> Self {
        Self {
            node: node.suf_suc_node.clone(),
            intervals: Default::default(),
            children: Default::default(),
        }
    }
}

#[derive(Debug, Deref)]
pub(crate) struct SufSucCentroidTree {
    nodes: TypedVec<CentroidId, CentroidNode>,
}

#[derive(Debug, Deref)]
pub(crate) struct SufSucCentroidTrees {
    trees: TypedVec<ForestNodeId, SufSucCentroidTree>,
}

impl SufSucCentroidTrees {
    pub fn new(node_set: &SufSucNodeSet, forest: &SucForest) -> Self {
        Self {
            trees: forest
                .keys()
                .map(|i| SufSucCentroidTree::new(i, node_set, forest))
                .collect(),
        }
    }
}

typed_vec_index!(SubTreeNodeId, u16);

#[derive(Clone, Copy, Debug, Deref)]
struct SubTreeNodeRef<'a> {
    #[deref]
    forest_node: &'a SucNode,
    suf_suc_node: &'a SufSucNode,
}

#[derive(Debug, Deref)]
struct SubTreeNode<'a> {
    #[deref]
    node: SubTreeNodeRef<'a>,
    parent: Option<SubTreeNodeId>,
    children: SmallVec<[SubTreeNodeId; NUM_INLINE_FOREST_NODES]>,
    size: u16,
}

impl SufSucCentroidTree {
    pub fn new(start: ForestNodeId, node_set: &SufSucNodeSet, forest: &SucForest) -> Self {
        if start == FOREST_VIRTUAL_ROOT {
            return Self {
                nodes: TypedVec::with_capacity(0),
            };
        }

        let mut subtree = {
            let mut pool = Vec::with_capacity(NUM_INLINE_FOREST_NODES);
            let mut cursor = start;
            while cursor != FOREST_VIRTUAL_ROOT {
                let forest_node = &forest[cursor];
                let suf_suc_node = &node_set[cursor];
                pool.push(SubTreeNodeRef {
                    forest_node,
                    suf_suc_node,
                });
                cursor = node_set.suffix_parent[cursor];
            }
            pool.reverse();
            debug_assert!(pool[0].parent == FOREST_VIRTUAL_ROOT);

            let mut forest_to_node_id = BTreeMap::new();

            let mut nodes = TypedVec::<SubTreeNodeId, _>::with_capacity(pool.len());
            for node in pool {
                let forest_id = node.suf_suc_node.forest_id;
                if node.parent == FOREST_VIRTUAL_ROOT {
                    let id = nodes.push(SubTreeNode {
                        node,
                        parent: None,
                        children: Default::default(),
                        size: 1,
                    });
                    forest_to_node_id.insert(forest_id, id);
                } else {
                    let parent = forest_to_node_id[&node.parent];
                    let id = nodes.push(SubTreeNode {
                        node,
                        parent: Some(parent),
                        children: Default::default(),
                        size: 1,
                    });
                    forest_to_node_id.insert(forest_id, id);
                    nodes[parent].children.push(id);
                }
            }

            for id in nodes.keys().rev() {
                let node = &nodes[id];
                if let Some(parent) = node.parent {
                    nodes[parent].size += node.size;
                    debug_assert!(parent < id);
                } else {
                    debug_assert!(id == SubTreeNodeId::ZERO);
                }
            }

            nodes
        };

        let mut roots = vec![(SubTreeNodeId::ZERO, None::<CentroidId>)];
        let mut centroids = TypedVec::<CentroidId, _>::with_capacity(subtree.len().as_usize());

        while let Some((root_id, parent_centroid)) = roots.pop() {
            let half_size = subtree[root_id].size / 2;
            let next_large_subtree = |id: SubTreeNodeId| -> Option<(usize, SubTreeNodeId)> {
                subtree[id]
                    .children
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(_, c)| subtree[*c].size > half_size)
            };

            let centroid = if let Some(child) = next_large_subtree(root_id) {
                let mut large_child = (root_id, child.0, child.1);
                while let Some(child) = next_large_subtree(large_child.2) {
                    large_child = (large_child.2, child.0, child.1);
                }

                let (parent, child_idx, centroid) = large_child;
                subtree[parent].children.swap_remove(child_idx);
                subtree[centroid].parent = None;

                let centroid_size = subtree[centroid].size;
                let mut parent = Some(parent);
                while let Some(parent_id) = parent {
                    subtree[parent_id].size -= centroid_size;
                    parent = subtree[parent_id].parent;
                }

                centroid
            } else {
                root_id
            };

            debug_assert!(
                subtree[centroid]
                    .children
                    .iter()
                    .all(|&i| subtree[i].size <= half_size)
            );

            let id = centroids.push(CentroidNode::new(*subtree[centroid]));
            if let Some(parent) = parent_centroid {
                let parent_node = &mut centroids[parent];
                parent_node
                    .intervals
                    .push(subtree[root_id].suf_suc_node.valid_range);
                parent_node.children.push(id);
            }
            for c in std::mem::take(&mut subtree[centroid].children) {
                let child = &mut subtree[c];
                child.parent = None;
                subtree[centroid].size -= child.size;
                roots.push((c, Some(id)));
            }
            if centroid != root_id {
                roots.push((root_id, None));
                debug_assert!(subtree[root_id].size <= half_size);
            }
        }

        #[cfg(debug_assertions)]
        {
            for node in subtree {
                debug_assert!(node.size == 1 && node.parent.is_none() && node.children.is_empty());
            }
        }

        for id in centroids.keys() {
            let mut order = Vec::from_iter(0..centroids[id].children.len());
            order.sort_unstable_by_key(|&i| centroids[id].intervals[i].0);
            let children = order
                .iter()
                .copied()
                .map(|i| centroids[id].children[i])
                .collect();
            centroids[id].children.clone_from(&children);
            let intervals = order
                .iter()
                .copied()
                .map(|i| centroids[id].intervals[i])
                .collect();
            centroids[id].intervals.clone_from(&intervals);
        }

        Self { nodes: centroids }
    }

    #[inline]
    pub fn search<F: Fn(usize) -> ForestNodeId>(&self, skip_to: F) -> ForestNodeId {
        let len = self.len();
        let to_parent = |node: CentroidId| {
            Some(node.next())
                .filter(|&parent| parent < len && self[parent].depth < self[node].depth)
        };

        let next_subtree = |node_id: CentroidId| {
            let node = &self[node_id];
            if node.children.is_empty() {
                return None;
            }
            let val = skip_to(node.skip_len as _);
            match node.intervals.binary_search_by_key(&val, |&(l, _)| l) {
                Ok(i) => Some(node.children[i]),
                Err(i) => {
                    if i == 0 {
                        return None;
                    }
                    let (_, r) = node.intervals[i - 1];
                    if val >= r {
                        return None;
                    }
                    Some(node.children[i - 1])
                }
            }
        };

        let mut current = CentroidId::ZERO;

        loop {
            if !self[current].verify(&skip_to) {
                if let Some(parent) = to_parent(current) {
                    current = parent;
                    continue;
                } else {
                    debug_assert!(false);
                    break;
                }
            }
            if let Some(child) = next_subtree(current) {
                current = child;
            } else {
                break;
            }
        }
        self[current].forest_id
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Dictionary, NormalizedDict, Vocab,
        aho_corasick::ACAutomaton,
        centroid::SufSucCentroidTrees,
        successor::{FOREST_VIRTUAL_ROOT, SucForest},
        suf_suc::SufSucNodeSet,
    };

    fn centroid_case(rules: &[(&str, &str)]) {
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

        let dict = Dictionary::new_from_token_pair(vocab, rules.iter().copied()).unwrap();
        let automaton = ACAutomaton::new(&dict);
        let normalized = NormalizedDict::new_in_bytes(dict.clone());
        let forest = SucForest::new(&normalized);
        let node_set = SufSucNodeSet::new(&forest, &automaton);
        let trees = SufSucCentroidTrees::new(&node_set, &forest);

        for (id, tree) in trees.enumerate() {
            if id == FOREST_VIRTUAL_ROOT {
                continue;
            }
            let token = &dict[forest[id].token_id];
            assert!(
                dict.tokens.iter().filter(|i| token.ends_with(i)).count() == tree.len().as_usize()
            );
            for u in tree.keys() {
                let v = u.next();
                if v >= tree.len() {
                    continue;
                }
                assert!(tree[u].forest_id != tree[v].forest_id);
                let is_parent = {
                    let mut w = forest[tree[u].forest_id].parent;
                    while w != FOREST_VIRTUAL_ROOT && w != tree[v].forest_id {
                        w = forest[w].parent;
                    }
                    w == tree[v].forest_id
                };
                dbg!(u, &tree[u]);
                dbg!(v, &tree[v]);
                assert!(is_parent ^ (tree[v].depth >= tree[u].depth));
            }
        }
    }

    #[test]
    fn test_centroid() {
        centroid_case(&[
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
        ]);
        centroid_case(&[
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
        ]);
    }
}
