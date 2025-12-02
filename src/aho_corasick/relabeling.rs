use crate::{
    aho_corasick::{AC_NODE_ROOT, ACNodeId},
    typed_vec::TypedVec,
};

#[derive(Debug)]
pub(super) struct Relabeling {
    order: TypedVec<ACNodeId, ACNodeId>,
    rank: TypedVec<ACNodeId, ACNodeId>,
}

impl Relabeling {
    pub fn new(order: TypedVec<ACNodeId, ACNodeId>) -> Self {
        debug_assert!(!order.is_empty());
        let mut rank = TypedVec::new_with(AC_NODE_ROOT, order.len());
        for (new_id, old_id) in order.enumerate_copied() {
            debug_assert!(
                (old_id == AC_NODE_ROOT && new_id == AC_NODE_ROOT)
                    || (old_id != AC_NODE_ROOT && rank[old_id] == AC_NODE_ROOT)
            );
            rank[old_id] = new_id;
        }
        Self { order, rank }
    }

    pub fn apply_to_typed_vec<T: Default>(
        &self,
        mut seq: TypedVec<ACNodeId, T>,
    ) -> TypedVec<ACNodeId, T> {
        seq.keys()
            .map(|i| std::mem::take(&mut seq[self.order[i]]))
            .collect()
    }

    pub fn apply_to_iter_mut<'a>(&self, iter: impl IntoIterator<Item = &'a mut ACNodeId>) {
        for slot in iter {
            *slot = self.rank[*slot];
        }
    }
}
