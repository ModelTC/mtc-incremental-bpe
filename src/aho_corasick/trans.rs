use crate::{
    aho_corasick::{AC_NODE_ROOT, ACNodeId, ACSuffixLinkTree, ACTrie},
    typed_vec::{TypedVec, typed_vec_index},
};

const ALPHABET_SIZE: usize = 1 << 8;
const TRANS_TILE: usize = 1 << 4;

const _: () = {
    assert!(TRANS_TILE * TRANS_TILE == ALPHABET_SIZE);
};

#[inline(always)]
fn split_byte(key: u8) -> (usize, usize) {
    let high = (key >> 4) as usize;
    let low = (key & 0xf) as usize;
    (high, low)
}

typed_vec_index!(TransId, u32);

const DEFAULT_TRANS_ID: TransId = TransId::ZERO;

#[derive(Debug)]
pub(crate) struct ACTransTable {
    trans: TypedVec<TransId, [ACNodeId; TRANS_TILE]>,
    base: TypedVec<ACNodeId, [TransId; TRANS_TILE]>,
}

#[derive(Debug)]
struct TransTableBuilder {
    table: ACTransTable,
    owner: TypedVec<TransId, ACNodeId>,
}

impl TransTableBuilder {
    fn new(num_nodes: ACNodeId) -> Self {
        let max_trans = TransId::from(num_nodes.as_usize());

        let mut trans = TypedVec::with_capacity(max_trans);
        let mut owner = TypedVec::with_capacity(max_trans);
        trans.push([AC_NODE_ROOT; TRANS_TILE]);
        owner.push(ACNodeId::MAX);

        let base = TypedVec::new_with([DEFAULT_TRANS_ID; TRANS_TILE], num_nodes);

        Self {
            table: ACTransTable { trans, base },
            owner,
        }
    }

    #[inline(always)]
    fn update_trans(&mut self, from: ACNodeId, to: ACNodeId, key: u8) {
        let (high, low) = split_byte(key);
        let mut trans_id = self.table.base[from][high];
        if self.owner[trans_id] != from {
            trans_id = self.table.trans.push(self.table.trans[trans_id]);
            self.owner.push(from);
            self.table.base[from][high] = trans_id;
        }
        self.table.trans[trans_id][low] = to;
    }

    #[inline(always)]
    fn duplicate(&mut self, current: ACNodeId, parent: ACNodeId) {
        self.table.base[current] = self.table.base[parent];
    }
}

impl ACTransTable {
    pub fn new(trie: &ACTrie, suffix: &ACSuffixLinkTree) -> Self {
        let mut builder = TransTableBuilder::new(trie.len());

        for current in trie.bfs() {
            let parent = suffix[current];
            builder.duplicate(current, parent);
            for (child, key) in trie.children(current) {
                builder.update_trans(current, child, key);
            }
        }

        builder.table
    }

    #[inline(always)]
    pub fn num_of_nodes(&self) -> ACNodeId {
        self.base.len()
    }

    #[inline(always)]
    pub fn get(&self, node_id: ACNodeId, key: u8) -> ACNodeId {
        let (high, low) = split_byte(key);
        self.trans[self.base[node_id][high]][low]
    }

    #[inline(always)]
    pub fn feed<B: AsRef<[u8]>>(&self, mut node: ACNodeId, bytes: B) -> ACNodeId {
        for &byte in bytes.as_ref() {
            node = self.get(node, byte);
        }
        node
    }
}
