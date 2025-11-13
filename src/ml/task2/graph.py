import torch
import numpy as np
from collections import defaultdict

import numpy as np
from collections import defaultdict

class UserItemNetwork:
    def __init__(self, df):

        users = df["user"].unique()
        items = df["item"].unique()

        # id ↔ idx 
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.item2idx = {it: i for i, it in enumerate(items)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.idx2item = {i: it for it, i in self.item2idx.items()}

        self.num_users = len(self.user2idx)
        self.num_items = len(self.item2idx)

        # adjacency list
        self.user_to_items = defaultdict(list)
        self.item_to_users = defaultdict(list)

        for _, row in df.iterrows():
            u_idx = self.user2idx[row["user"]]
            i_idx = self.item2idx[row["item"]]
            self.user_to_items[u_idx].append(i_idx)
            self.item_to_users[i_idx].append(u_idx)

    # 이 유저가 train에서 2/3 했던 item idx들
    def get_seed_items(self, user_raw_id):
        if user_raw_id not in self.user2idx:
            return []
        u_idx = self.user2idx[user_raw_id]
        return self.user_to_items.get(u_idx, [])

    def get_random_item_from_user(self, u_idx):
        items = self.user_to_items.get(u_idx)
        if not items:
            return None
        return np.random.choice(items)

    def get_random_user_from_item(self, i_idx):
        users = self.item_to_users.get(i_idx)
        if not users:
            return None
        return np.random.choice(users)


def pixie_recommend(
    graph: UserItemNetwork,
    user_raw_id: int,
    n_steps: int = 2000,
    alpha: float = 0.5,
    topk: int = 50,
):
    """
    graph: 위에서 만든 UserItemNetwork
    user_raw_id: 추천해줄 user (raw id, df에 있던 그대로)
    n_steps: random walk 총 스텝
    alpha: restart 확률
    topk: 몇 개 추천할지
    """

    # 1) seed 아이템들: 이 유저가 2/3로 상호작용한 아이템
    seed_item_idxs = graph.get_seed_items(user_raw_id)
    if not seed_item_idxs:
        # 이 유저가 train에 없거나 2/3가 없으면 그냥 빈 리스트
        return []

    # visit count 저장
    visit_count = defaultdict(int)

    # 2) walk 시작 아이템을 seed에서 하나 뽑음
    # seed 간에 가중치를 줄 거면 여기서 weight를 줄 수도 있음
    cur_item = np.random.choice(seed_item_idxs)

    for _ in range(n_steps):
        # item -> user
        u = graph.get_random_user_from_item(cur_item)
        if u is None:
            # 이 아이템에 연결된 유저가 없으면 seed로 리셋
            cur_item = np.random.choice(seed_item_idxs)
            continue

        # user -> item
        nxt_item = graph.get_random_item_from_user(u)
        if nxt_item is None:
            # 이 유저가 가진 아이템이 없으면 seed로 리셋
            cur_item = np.random.choice(seed_item_idxs)
            continue

        # 방문 기록
        visit_count[nxt_item] += 1

        # restart?
        if np.random.rand() < alpha:
            cur_item = np.random.choice(seed_item_idxs)
        else:
            cur_item = nxt_item

    seen = set(seed_item_idxs)
    candidates = []
    for i_idx, cnt in visit_count.items():
        if i_idx in seen:
            continue
        candidates.append((i_idx, cnt))

    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[:topk]

    recommended_item_ids = [graph.idx2item[i_idx] for i_idx, _ in candidates]
    return recommended_item_ids