from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from Bio import SeqIO
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sequence_models.collaters import SimpleCollater, StructureCollater, PROTEIN_ALPHABET
import esm
import esm.inverse_folding
from sequence_models.pdb_utils import parse_PDB, process_coords


def read_sequence(fasta_file):
    """读取单条fasta序列"""
    records = SeqIO.parse(fasta_file, "fasta")
    for each in records:
        return str(each.seq)


class BaseMutationDataset(Dataset, ABC):
    """突变数据集基类，提取公共逻辑"""

    def __init__(
        self,
        data_path,
        score_column,
        max_seq_length,
        require_pdb=False,
        dataset_list_txt=None,
        data_root="dataset/data",
        zscore_norm=False,
    ):
        self.dataset_list_txt = Path(dataset_list_txt) if dataset_list_txt else None
        self.data_root = Path(data_root)
        self.data_path = Path(data_path) if self.dataset_list_txt is None else self.data_root
        self.fasta_folder = self.data_path / "fasta"
        self.mutant_folder = self.data_path / "mutant"
        self.pdb_folder = self.data_path / "pdb"
        self.require_pdb = require_pdb

        # 验证必要的文件夹存在
        assert self.fasta_folder.exists() and self.mutant_folder.exists(), \
            f"Data path {self.data_path} does not contain fasta and mutant folders"
        if self.require_pdb:
            assert self.pdb_folder.exists(), \
                f"Data path {self.data_path} does not contain pdb folder"
        if self.dataset_list_txt is not None:
            assert self.dataset_list_txt.exists(), \
                f"Dataset list txt not found: {self.dataset_list_txt}"

        self.score_column = score_column
        self.max_seq_length = max_seq_length
        self.zscore_norm = zscore_norm
        self.data = self._load()

    @abstractmethod
    def _get_vocab_size(self) -> int:
        """返回词表大小"""
        pass

    @abstractmethod
    def _get_token_idx(self, aa: str) -> int:
        """将氨基酸转为索引"""
        pass

    def _parse_mutants(self, df, seq_length):
        """解析突变数据，构建 landscape 和 mask"""
        vocab_size = self._get_vocab_size()
        landscape = torch.zeros((seq_length, vocab_size))
        mask = torch.zeros((seq_length, vocab_size))

        for _, row in df.iterrows():
            # 跳过多位点突变
            if ":" in row["mutant"] or ";" in row["mutant"]:
                continue
            mutant = row["mutant"]
            wt, pos, mut = mutant[0], int(mutant[1:-1]) - 1, mutant[-1]
            token_idx = self._get_token_idx(mut)
            if token_idx is not None:
                landscape[pos, token_idx] = row[self.score_column]
                mask[pos, token_idx] = 1

        return landscape, mask

    def _process_item(self, seq, mutant_file, fasta_file, pdb_file, df):
        """处理单条数据的特殊逻辑，子类可覆盖"""
        landscape, mask = self._parse_mutants(df, len(seq))
        return {"seq": seq, "landscape": landscape, "mask": mask.bool()}

    def _get_mutant_files(self):
        """根据配置获取待加载的 mutant csv 文件列表。"""
        if self.dataset_list_txt is None:
            return list(self.mutant_folder.glob("*.csv"))

        names = []
        with open(self.dataset_list_txt, "r") as f:
            for line in f:
                name = line.strip()
                if name and not name.startswith("#"):
                    names.append(name)

        return [self.mutant_folder / f"{name}.csv" for name in names]

    def _load(self):
        """加载数据"""
        data = []
        mutant_files = self._get_mutant_files()

        for mutant_file in mutant_files:
            fasta_file = self.fasta_folder / mutant_file.with_suffix(".fasta").name
            pdb_file = self.pdb_folder / mutant_file.with_suffix(".pdb").name if self.require_pdb else None

            assert mutant_file.exists(), f"Mutant file not found: {mutant_file}"
            assert fasta_file.exists(), f"Fasta file not found: {fasta_file}"
            if self.require_pdb:
                assert pdb_file is not None and pdb_file.exists(), f"PDB file not found: {pdb_file}"

            seq = read_sequence(fasta_file)
            if self.max_seq_length and len(seq) > self.max_seq_length:
                continue

            df = pd.read_csv(mutant_file)
            if self.score_column == "ddG":
                df[self.score_column] = df[self.score_column] * -1
            if self.zscore_norm:
                single_site_mask = ~df["mutant"].astype(str).str.contains("[:;]", na=False)
                scores = pd.to_numeric(df.loc[single_site_mask, self.score_column], errors="coerce")
                valid_scores = scores.dropna()
                if not valid_scores.empty:
                    mean = valid_scores.mean()
                    std = valid_scores.std(ddof=0)
                    if std > 0:
                        normalized_scores = (scores - mean) / std
                    else:
                        normalized_scores = scores - mean
                    df.loc[single_site_mask, self.score_column] = normalized_scores

            item = self._process_item(seq, mutant_file, fasta_file, pdb_file, df)
            if item is not None:
                data.append(item)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @abstractmethod
    def collate_fn(self, batch):
        """批处理函数，子类必须实现"""
        pass

    def get_dataloader(self, batch_size=4, shuffle=False, **kwargs):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            **kwargs
        )


class ESM2Dataset(BaseMutationDataset):
    """ESM2模型的突变数据集"""

    def __init__(
        self,
        data_path,
        score_column,
        max_seq_length,
        model_path,
        dataset_list_txt=None,
        data_root="dataset/data",
        zscore_norm=False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.vocab = self.tokenizer.get_vocab()
        super().__init__(
            data_path,
            score_column,
            max_seq_length,
            require_pdb=False,
            dataset_list_txt=dataset_list_txt,
            data_root=data_root,
            zscore_norm=zscore_norm,
        )

    def _get_vocab_size(self):
        return len(self.vocab)

    def _get_token_idx(self, aa):
        return self.vocab.get(aa)

    def collate_fn(self, batch):
        seqs = [x["seq"] for x in batch]
        landscapes = [x["landscape"] for x in batch]
        masks = [x["mask"] for x in batch]
        tokenized = self.tokenizer(seqs, return_tensors="pt", padding=True)
        landscapes = pad_sequence(landscapes, batch_first=True, padding_value=0)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "landscape": landscapes,
            "mask": masks.bool(),
        }


class ProSSTDataset(BaseMutationDataset):
    """ProSST 模型的突变数据集"""

    def __init__(
        self,
        data_path,
        score_column,
        max_seq_length,
        model_path,
        prosst_data_path="dataset/data/prosst",
        dataset_list_txt=None,
        data_root="dataset/data",
        zscore_norm=False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.vocab = self.tokenizer.get_vocab()
        self.prosst_data_path = Path(prosst_data_path)
        assert self.prosst_data_path.exists(), f"ProSST structure folder not found: {self.prosst_data_path}"
        super().__init__(
            data_path,
            score_column,
            max_seq_length,
            require_pdb=False,
            dataset_list_txt=dataset_list_txt,
            data_root=data_root,
            zscore_norm=zscore_norm,
        )

    def _get_vocab_size(self):
        return len(self.vocab)

    def _get_token_idx(self, aa):
        return self.vocab.get(aa)

    def _process_item(self, seq, mutant_file, fasta_file, pdb_file, df):
        landscape, mask = self._parse_mutants(df, len(seq))
        prosst_file = self.prosst_data_path / fasta_file.name
        assert prosst_file.exists(), f"ProSST structure file not found: {prosst_file}"
        ss_tokens = read_sequence(prosst_file).split(",")
        ss_ids = [int(token) + 3 for token in ss_tokens if token != ""]
        if len(ss_ids) != len(seq):
            raise ValueError(
                f"ProSST structure length mismatch: {fasta_file.name} seq={len(seq)} ss={len(ss_ids)}"
            )
        # ProSST convention: 1/2 are bos/eos, structure ids are offset by +3.
        ss_input_ids = torch.tensor([1, *ss_ids, 2], dtype=torch.long)
        return {
            "seq": seq,
            "ss_input_ids": ss_input_ids,
            "landscape": landscape,
            "mask": mask.bool(),
        }

    def collate_fn(self, batch):
        seqs = [x["seq"] for x in batch]
        landscapes = [x["landscape"] for x in batch]
        masks = [x["mask"] for x in batch]
        ss_input_ids = [x["ss_input_ids"] for x in batch]
        tokenized = self.tokenizer(seqs, return_tensors="pt", padding=True)
        landscapes = pad_sequence(landscapes, batch_first=True, padding_value=0)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        ss_input_ids = pad_sequence(ss_input_ids, batch_first=True, padding_value=0)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "ss_input_ids": ss_input_ids,
            "landscape": landscapes,
            "mask": masks.bool(),
        }


class ESMIFDataset(BaseMutationDataset):
    """ESM-IF模型的突变数据集"""

    def __init__(
        self,
        data_path,
        score_column,
        max_seq_length,
        dataset_list_txt=None,
        data_root="dataset/data",
        zscore_norm=False,
    ):
        _, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.batch_converter = esm.inverse_folding.util.CoordBatchConverter(self.alphabet)
        super().__init__(
            data_path,
            score_column,
            max_seq_length,
            require_pdb=True,
            dataset_list_txt=dataset_list_txt,
            data_root=data_root,
            zscore_norm=zscore_norm,
        )

    def _get_vocab_size(self):
        return len(self.alphabet)

    def _get_token_idx(self, aa):
        return self.alphabet.get_idx(aa)

    def _process_item(self, seq, mutant_file, fasta_file, pdb_file, df):
        coords, native_seq = esm.inverse_folding.util.load_coords(str(pdb_file), "A")
        assert len(native_seq) == len(seq), \
            f"Sequence length mismatch: {fasta_file} vs {pdb_file}"

        landscape, mask = self._parse_mutants(df, len(seq))
        return {
            "seq": seq,
            "coords": coords,
            "landscape": landscape,
            "mask": mask.bool(),
            "confidence": None
        }

    def collate_fn(self, batch):
        seqs = [x["seq"] for x in batch]
        landscapes = [x["landscape"] for x in batch]
        masks = [x["mask"] for x in batch]
        coords_list = [x["coords"] for x in batch]
        confidence_list = [x["confidence"] for x in batch]

        landscapes = pad_sequence(landscapes, batch_first=True, padding_value=0.0)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)

        coords, confidence, _, tokens, padding_mask = self.batch_converter(
            list(zip(coords_list, confidence_list, seqs)))
        return {
            "input_ids": tokens,
            "coords": coords,
            "confidence": confidence,
            "attention_mask": padding_mask,
            "landscape": landscapes,
            "mask": masks,
        }

    def get_dataloader(self, batch_size=1, shuffle=False, **kwargs):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            **kwargs
        )


class ProteinMPNNDataset(BaseMutationDataset):
    """ProteinMPNN模型的突变数据集"""

    ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

    def __init__(
        self,
        data_path,
        score_column,
        max_seq_length,
        dataset_list_txt=None,
        data_root="dataset/data",
        zscore_norm=False,
    ):
        super().__init__(
            data_path,
            score_column,
            max_seq_length,
            require_pdb=True,
            dataset_list_txt=dataset_list_txt,
            data_root=data_root,
            zscore_norm=zscore_norm,
        )

    def _get_vocab_size(self):
        return len(self.ALPHABET)

    def _get_token_idx(self, aa):
        if aa in self.ALPHABET:
            return self.ALPHABET.index(aa)
        return None

    def _parse_pdb(self, pdb_file, chain='A'):
        """解析PDB文件，提取N, CA, C, O原子坐标"""
        alpha_3_to_1 = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
            'MSE': 'M',  # Selenomethionine treated as Met
        }
        atoms = ['N', 'CA', 'C', 'O']
        xyz = {}
        seq = {}

        with open(pdb_file, 'r') as f:
            for line in f:
                if line[:6] == "HETATM" and line[17:20] == "MSE":
                    line = "ATOM  " + line[6:]
                if line[:4] == "ATOM":
                    ch = line[21]
                    if ch != chain:
                        continue
                    atom = line[12:16].strip()
                    resi = line[17:20].strip()
                    resn = int(line[22:26].strip())
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    if resn not in xyz:
                        xyz[resn] = {}
                    if atom in atoms:
                        xyz[resn][atom] = [x, y, z]
                    if resn not in seq:
                        seq[resn] = alpha_3_to_1.get(resi, 'X')

        # 转换为有序数组
        sorted_resns = sorted(xyz.keys())
        coords = []
        sequence = []
        for resn in sorted_resns:
            atom_coords = []
            for atom in atoms:
                if atom in xyz[resn]:
                    atom_coords.append(xyz[resn][atom])
                else:
                    atom_coords.append([float('nan')] * 3)
            coords.append(atom_coords)
            sequence.append(seq.get(resn, 'X'))

        return torch.tensor(coords, dtype=torch.float32), ''.join(sequence)

    def _process_item(self, seq, mutant_file, fasta_file, pdb_file, df):
        # 解析PDB获取坐标
        coords, native_seq = self._parse_pdb(str(pdb_file))
        assert native_seq == seq, f"Sequence mismatch: {fasta_file} ({len(seq)}) vs {pdb_file} ({len(native_seq)})"

        landscape, mask = self._parse_mutants(df, len(seq))

        # 序列转索引
        S = torch.tensor([self.ALPHABET.index(aa) if aa in self.ALPHABET else self.ALPHABET.index('X')
                          for aa in seq], dtype=torch.long)

        # residue_idx: 残基索引 (0-indexed)
        residue_idx = torch.arange(len(seq), dtype=torch.long)

        # chain_encoding: 链编码，单链设为1
        chain_encoding = torch.ones(len(seq), dtype=torch.long)

        return {
            "seq": seq,
            "X": coords,  # [L, 4, 3] - N, CA, C, O
            "S": S,  # [L] - 序列索引
            "residue_idx": residue_idx,  # [L]
            "chain_encoding": chain_encoding,  # [L]
            "landscape": landscape,
            "mask": mask.bool(),
        }

    def collate_fn(self, batch):
        B = len(batch)
        lengths = [len(b["seq"]) for b in batch]
        L_max = max(lengths)

        # 初始化张量
        X = torch.zeros((B, L_max, 4, 3), dtype=torch.float32)
        S = torch.zeros((B, L_max), dtype=torch.long)
        attention_mask = torch.zeros((B, L_max), dtype=torch.float32)
        residue_idx = torch.zeros((B, L_max), dtype=torch.long)
        chain_encoding = torch.zeros((B, L_max), dtype=torch.long)
        landscapes = torch.zeros((B, L_max, len(self.ALPHABET)), dtype=torch.float32)
        masks = torch.zeros((B, L_max, len(self.ALPHABET)), dtype=torch.bool)

        for i, b in enumerate(batch):
            l = lengths[i]
            X[i, :l] = b["X"]
            S[i, :l] = b["S"]
            attention_mask[i, :l] = 1.0
            residue_idx[i, :l] = b["residue_idx"]
            chain_encoding[i, :l] = b["chain_encoding"]
            landscapes[i, :l] = b["landscape"]
            masks[i, :l] = b["mask"]

        # 处理NaN值
        X = torch.nan_to_num(X, nan=0.0)

        return {
            "X": X,  # [B, L, 4, 3]
            "input_ids": S,  # [B, L]
            "attention_mask": attention_mask,  # [B, L]
            "residue_idx": residue_idx,  # [B, L]
            "chain_encoding": chain_encoding,  # [B, L]
            "landscape": landscapes,  # [B, L, 21]
            "mask": masks,  # [B, L, 21]
        }


class MIFSTDataset(BaseMutationDataset):
    """MIFST模型的突变数据集"""

    def __init__(
        self,
        data_path,
        score_column,
        max_seq_length,
        dataset_list_txt=None,
        data_root="dataset/data",
        zscore_norm=False,
    ):
        self.alphabet = PROTEIN_ALPHABET
        sequence_collater = SimpleCollater(PROTEIN_ALPHABET, pad=True)
        self.structure_collater = StructureCollater(sequence_collater, n_connections=30)
        super().__init__(
            data_path,
            score_column,
            max_seq_length,
            require_pdb=True,
            dataset_list_txt=dataset_list_txt,
            data_root=data_root,
            zscore_norm=zscore_norm,
        )

    def _get_vocab_size(self):
        return len(self.alphabet)

    def _get_token_idx(self, aa):
        if aa in self.alphabet:
            return self.alphabet.index(aa)
        return None

    def _process_item(self, seq, mutant_file, fasta_file, pdb_file, df):
        # 解析结构
        coords, native_seq, _ = parse_PDB(str(pdb_file))
        coords_dict = {
            'N': coords[:, 0],
            'CA': coords[:, 1],
            'C': coords[:, 2]
        }
        dist, omega, theta, phi = process_coords(coords_dict)

        assert native_seq == seq, f"Sequence mismatch: {fasta_file} vs {pdb_file}"

        landscape, mask = self._parse_mutants(df, len(seq))

        return {
            "seq": seq,
            "dist": dist,
            "omega": omega,
            "theta": theta,
            "phi": phi,
            "landscape": landscape,
            "mask": mask.bool(),
        }

    def collate_fn(self, batch):
        landscapes = [x["landscape"] for x in batch]
        masks = [x["mask"] for x in batch]

        mifst_batch = [[
            x["seq"],
            torch.tensor(x["dist"], dtype=torch.float),
            torch.tensor(x["omega"], dtype=torch.float),
            torch.tensor(x["theta"], dtype=torch.float),
            torch.tensor(x["phi"], dtype=torch.float)
        ] for x in batch]

        input_ids, nodes, edges, connections, edge_mask = self.structure_collater(mifst_batch)
        landscapes = pad_sequence(landscapes, batch_first=True, padding_value=0.0)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "nodes": nodes,
            "edges": edges,
            "connections": connections,
            "edge_mask": edge_mask,
            "landscape": landscapes,
            "mask": masks,
        }


class ProteinMPNNESMDataset(BaseMutationDataset):
    """ProteinMPNN + ESM2 集成模型的突变数据集
    
    同时提供 ProteinMPNN 和 ESM2 所需的输入数据。
    landscape 使用 ProteinMPNN 的词表 (21 维)。
    """

    ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'  # ProteinMPNN 词表

    def __init__(
        self,
        data_path,
        score_column,
        max_seq_length,
        esm_model_path="facebook/esm2_t33_650M_UR50D",
        dataset_list_txt=None,
        data_root="dataset/data",
        zscore_norm=False,
    ):
        # 初始化 ESM tokenizer
        self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_path)
        super().__init__(
            data_path,
            score_column,
            max_seq_length,
            require_pdb=True,
            dataset_list_txt=dataset_list_txt,
            data_root=data_root,
            zscore_norm=zscore_norm,
        )

    def _get_vocab_size(self):
        # 使用 ProteinMPNN 词表大小
        return len(self.ALPHABET)

    def _get_token_idx(self, aa):
        if aa in self.ALPHABET:
            return self.ALPHABET.index(aa)
        return None

    def _parse_pdb(self, pdb_file, chain='A'):
        """解析PDB文件，提取N, CA, C, O原子坐标"""
        alpha_3_to_1 = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
            'MSE': 'M',  # Selenomethionine treated as Met
        }
        atoms = ['N', 'CA', 'C', 'O']
        xyz = {}
        seq = {}

        with open(pdb_file, 'r') as f:
            for line in f:
                if line[:6] == "HETATM" and line[17:20] == "MSE":
                    line = "ATOM  " + line[6:]
                if line[:4] == "ATOM":
                    ch = line[21]
                    if ch != chain:
                        continue
                    atom = line[12:16].strip()
                    resi = line[17:20].strip()
                    resn = int(line[22:26].strip())
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    if resn not in xyz:
                        xyz[resn] = {}
                    if atom in atoms:
                        xyz[resn][atom] = [x, y, z]
                    if resn not in seq:
                        seq[resn] = alpha_3_to_1.get(resi, 'X')

        # 转换为有序数组
        sorted_resns = sorted(xyz.keys())
        coords = []
        sequence = []
        for resn in sorted_resns:
            atom_coords = []
            for atom in atoms:
                if atom in xyz[resn]:
                    atom_coords.append(xyz[resn][atom])
                else:
                    atom_coords.append([float('nan')] * 3)
            coords.append(atom_coords)
            sequence.append(seq.get(resn, 'X'))

        return torch.tensor(coords, dtype=torch.float32), ''.join(sequence)

    def _process_item(self, seq, mutant_file, fasta_file, pdb_file, df):
        # 解析PDB获取坐标
        coords, native_seq = self._parse_pdb(str(pdb_file))
        assert native_seq == seq, f"Sequence mismatch: {fasta_file} ({len(seq)}) vs {pdb_file} ({len(native_seq)})"

        # 使用 ProteinMPNN 词表构建 landscape
        landscape, mask = self._parse_mutants(df, len(seq))

        # ProteinMPNN 序列索引
        S = torch.tensor([self.ALPHABET.index(aa) if aa in self.ALPHABET else self.ALPHABET.index('X')
                          for aa in seq], dtype=torch.long)

        # residue_idx: 残基索引 (0-indexed)
        residue_idx = torch.arange(len(seq), dtype=torch.long)

        # chain_encoding: 链编码，单链设为1
        chain_encoding = torch.ones(len(seq), dtype=torch.long)

        return {
            "seq": seq,
            "X": coords,  # [L, 4, 3] - N, CA, C, O
            "S": S,  # [L] - ProteinMPNN 序列索引
            "residue_idx": residue_idx,  # [L]
            "chain_encoding": chain_encoding,  # [L]
            "landscape": landscape,  # [L, 21] - ProteinMPNN 词表
            "mask": mask.bool(),
        }

    def collate_fn(self, batch):
        B = len(batch)
        seqs = [b["seq"] for b in batch]
        lengths = [len(seq) for seq in seqs]
        L_max = max(lengths)

        # === ProteinMPNN 输入 ===
        X = torch.zeros((B, L_max, 4, 3), dtype=torch.float32)
        S = torch.zeros((B, L_max), dtype=torch.long)
        attention_mask = torch.zeros((B, L_max), dtype=torch.float32)
        residue_idx = torch.zeros((B, L_max), dtype=torch.long)
        chain_encoding = torch.zeros((B, L_max), dtype=torch.long)
        landscapes = torch.zeros((B, L_max, len(self.ALPHABET)), dtype=torch.float32)
        masks = torch.zeros((B, L_max, len(self.ALPHABET)), dtype=torch.bool)

        for i, b in enumerate(batch):
            l = lengths[i]
            X[i, :l] = b["X"]
            S[i, :l] = b["S"]
            attention_mask[i, :l] = 1.0
            residue_idx[i, :l] = b["residue_idx"]
            chain_encoding[i, :l] = b["chain_encoding"]
            landscapes[i, :l] = b["landscape"]
            masks[i, :l] = b["mask"]

        # 处理NaN值
        X = torch.nan_to_num(X, nan=0.0)

        # === ESM2 输入 ===
        esm_tokenized = self.esm_tokenizer(seqs, return_tensors="pt", padding=True)

        return {
            # ProteinMPNN 字段
            "X": X,  # [B, L, 4, 3]
            "S": S,  # [B, L] - ProteinMPNN 序列索引
            "attention_mask": attention_mask,  # [B, L]
            "residue_idx": residue_idx,  # [B, L]
            "chain_encoding": chain_encoding,  # [B, L]
            # ESM2 字段
            "esm_input_ids": esm_tokenized["input_ids"],  # [B, L+2] 包含 sos/eos
            "esm_attention_mask": esm_tokenized["attention_mask"],  # [B, L+2]
            # 标签 (使用 ProteinMPNN 词表)
            "landscape": landscapes,  # [B, L, 21]
            "mask": masks,  # [B, L, 21]
        }
