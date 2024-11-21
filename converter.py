#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author:        Run yang

import os
import json
import zlib
import base64
import pandas as pd
from datasets import Dataset, concatenate_datasets
import multiprocessing
import logging
from tqdm import tqdm
from typing import List, Dict

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("processing.log")
    ]
)

room_columns=[
    'id', 
 'uid', # 用户ID
 'room_id', # 牌局ID
 'app_name', 
 'players_info', # 玩家信息
 'cards', #下发的牌型数据，seat=1玩家，seat=2玩家下家，seat=3玩家上家
 'base_bean',  # 底分，豆子
 'game_state',  # 游戏状态 1=开始 2=已结束
 'shuffle_times',  # 重新发牌次数
 'game_type',  # 牌局类型 1为非新手局，2为新手局
 'game_history', # 游戏操作记录(JSON:座次,操作,牌,道具)，解析命令 json.loads(zlib.decompress(base64.b64decode(his)))
 'landlord_seat', # 地主座位号
 'seat_multiply', # 倍数信息
 'winner_seat',  # 胜利方座位号
 'is_spring',  # 是否春天 0非春天 1春天 2反春天
 'left_cards', # 牌局结束后，各方剩余的牌(JSON)
 'card_type',  # 牌型枚举
 'first_seat',  # 默认第一个叫地主的人
 'global_multiply', # 全局倍数
 'room_level', # 房间难易 程度1=温暖局 2=普通局 3=难度局 4=必输局
 'over_type',  # 游戏结束类型 0-正常结算 1-中途结算
 'create_time', # 创建时间
 'update_time',  # 更新时间
 'current_multiply',  # 当前倍数-炸弹、王炸
 'sub_over_type',  # 1:主页或者上线强制结算 2:定时任务调用 3:内部接口手动CURL 调用
 'robot1_level',  # 机器人1等级 ,座位号2 真人玩家的下家
 'robot2_level',  # 机器人2等级 ,座位号3 真人玩家的上家
 'base_bean_level' ,  # 豆子等级
 'ticket_level', # 底分等级
    'is_win'  # 赢了1
]

# Constants
SP_TOKEN = {1: 98, 2: 99, 3: 100, "LL": 101, "R": 102,
            "bos": 103, "eos": 104, "boa": 105, "eoa": 106, "PAD": 107}
ING_TOKEN = -100

REAL_CARD_TO_ENV_CARD = {
    '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12,
    'K': 13, 'A': 14, '2': 17, 'X': 20, 'D': 30
}
CARD_TO_COLUMN = {
    3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
    11: 8, 12: 9, 13: 10, 14: 11, 17: 12, 20: 13, 30: 14
}


def get_total_chunks(input_file, chunk_size):
    with open(input_file, "r") as f:
        total_rows = sum(1 for _ in f) - 1  # 减去表头
    total_chunks = (total_rows + chunk_size - 1) // chunk_size  # 向上取整
    return total_chunks


def convert_episode(record):
    """Convert a single record into states, actions, trajectory, and targets."""
    try:
        winner_seat = record['winner_seat']
        landlord_seat = record['landlord_seat']
        global_multiply = record['global_multiply']
        states, targets = process_starting_hand_cards(record, landlord_seat)
        actions, history_targets = process_game_history(record, winner_seat, global_multiply)

        trajectory = states + actions
        targets.extend(history_targets)

        assert len(trajectory) == len(targets)
        return {"states": states, "actions": actions, "trajectory": trajectory, "targets": targets}
    except Exception as e:
        # logging.error(f"Error converting record: {e}")
        # logging.error(record)
        # import pdb;pdb.set_trace()
        raise

def process_starting_hand_cards(record, landlord_seat):
    cards_body = json.loads(record['cards'])
    for hand in cards_body['cards']:
        if hand['seat'] == landlord_seat:
            hand['card'].extend(cards_body['otherCards'])

    states = [SP_TOKEN["bos"]]
    for hand in cards_body['cards']:
        if hand['seat'] == landlord_seat:
            states.append(SP_TOKEN["LL"])
        states.append(SP_TOKEN[hand["seat"]])

        card_tokens = map_cards_to_tokens(hand['card'])
        states.extend(card_tokens)

        states.append(SP_TOKEN[hand['seat']])
        if hand['seat'] == landlord_seat:
            states.append(SP_TOKEN["LL"])
    states.append(SP_TOKEN["eos"])

    targets = [ING_TOKEN] * len(states)
    return states, targets

def process_game_history(record, winner_seat, global_multiply):
    history_body = json.loads(zlib.decompress(base64.b64decode(record['game_history'])))
    actions = []
    targets = []

    for action in history_body:
        if action['cards']:
            seat = action['seat']
            card_tokens = map_cards_to_tokens(action['cards'])

            actions.append(SP_TOKEN[seat])
            actions.append(SP_TOKEN["boa"])
            actions.extend(card_tokens)
            actions.append(SP_TOKEN["eoa"])

            targets.extend([ING_TOKEN] * (len(card_tokens) + 2))
            reward = get_reward(seat, winner_seat, global_multiply)
            targets.append(reward)

    return actions, targets

def map_cards_to_tokens(card_ids):
    card_values = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
    tokens = []
    for card_id in card_ids:
        if card_id <= 52:
            value_index = (card_id - 1) // 4
            card_value = card_values[value_index]
            env_card = REAL_CARD_TO_ENV_CARD[card_value]
            tokens.append(CARD_TO_COLUMN[env_card])
        elif card_id == 53:
            tokens.append(CARD_TO_COLUMN[REAL_CARD_TO_ENV_CARD['X']])
        elif card_id == 54:
            tokens.append(CARD_TO_COLUMN[REAL_CARD_TO_ENV_CARD['D']])
        else:
            raise ValueError(f"Invalid card ID: {card_id}")
    return tokens

def get_reward(seat_id, winner_seat, bomb_count):
    return 1.0  if seat_id == winner_seat else -1.0



def process_chunk(chunk: pd.DataFrame, output_dir: str, chunk_idx: int) -> None:
    """
    Process a single chunk of data, save results, and record processing statistics.

    Args:
        chunk (pd.DataFrame): The chunk of data to process.
        output_dir (str): Directory to save the processed chunk.
        chunk_idx (int): Index of the current chunk being processed.

    Returns:
        None: The function saves processed data directly to disk.
    """
    results: List[Dict] = []  # List to store successfully processed records
    success_count  = 0    # Counter for successfully processed rows
    error_count = 0      # Counter for rows with errors

    for row_idx, row in chunk.iterrows():
        record: Dict = row.to_dict()  # Convert the row to a dictionary
        try:
            # Convert record and add to results
            results.append(convert_episode(record))
            success_count += 1
        except Exception as e:
            error_count += 1
            # logging.warning(f"Error processing row {row_idx} in chunk {chunk_idx}: {e}")

    # Save processed records as a Hugging Face Dataset
    if results:
        dataset = Dataset.from_list(results)
        dataset.save_to_disk(os.path.join(output_dir, f"chunk_{chunk_idx}"))

    # Calculate success rate
    total_rows = success_count + error_count
    success_rate = (success_count / total_rows) * 100 if total_rows > 0 else 0.0

    # Log statistics for the current chunk
    logging.info(
        f"Chunk {chunk_idx} processed: "
        f"{success_count} successful, {error_count} errors, "
        f"success rate: {success_rate:.2f}%"
    )


def process_csv(input_file, output_dir, chunk_size=10000, num_workers=4):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Start to read csv file")
    reader = pd.read_csv(
        input_file,
        header=None,
        names=room_columns,
        chunksize=chunk_size,
        low_memory=False,
        # on_bad_lines='skip',
        nrows=5807270,
    )
    total_chunks = get_total_chunks(input_file, chunk_size)
    pool = multiprocessing.Pool(num_workers)
    logging.info(f"Total chunks to process: {total_chunks}")
    # for chunk_idx, chunk in enumerate(reader):
    #     process_chunk(chunk, output_dir, chunk_idx)
    with tqdm(total=total_chunks, desc="Processing Chunks", unit="chunk") as pbar:
        for chunk_idx, chunk in enumerate(reader):
            pool.apply_async(
                process_chunk,
                args=(chunk, output_dir, chunk_idx),
                callback=lambda _: pbar.update()
            )
        pool.close()
        pool.join()

def merge_datasets(output_dir, final_output_path):
    dataset_list = []
    for chunk_file in tqdm(sorted(os.listdir(output_dir)), desc="Merging Chunks"):
        if chunk_file.startswith("chunk_"):
            dataset = Dataset.load_from_disk(os.path.join(output_dir, chunk_file))
            dataset_list.append(dataset)
    final_dataset = concatenate_datasets(dataset_list)
    final_dataset.save_to_disk(final_output_path)

if __name__ == "__main__":
    input_csv = "/mnt/dhsys/doudizhu/data_room1101_part1.csv"
    output_chunks_dir = "/mnt/dhsys/doudizhu/processed_chunks-reward-shape"
    final_output_path = "/mnt/dhsys/doudizhu/final_dataset-reward-shape"
    
    os.makedirs(final_output_path, exist_ok=True)
    # Step 1: Process CSV in chunks
    process_csv(input_csv, output_chunks_dir, chunk_size=10000, num_workers=16)

    # Step 2: Merge chunks into a single Dataset
    merge_datasets(output_chunks_dir, final_output_path)

    logging.info(f"Processing complete. Final dataset saved at {final_output_path}")
