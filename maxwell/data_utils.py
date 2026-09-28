from .datasets import ESM2Dataset, ESMIFDataset, MIFSTDataset, ProteinMPNNDataset, ProteinMPNNESMDataset, ProSSTDataset


def _make_dataset(
    model_type,
    data_path,
    score_column,
    max_seq_length,
    model_path=None,
    esm_model_path=None,
    prosst_data_path="dataset/data/prosst",
    dataset_list_txt=None,
    data_root="dataset/data",
    zscore_norm=False,
):
    if model_type == "esm":
        return ESM2Dataset(
            data_path=data_path,
            score_column=score_column,
            max_seq_length=max_seq_length,
            model_path=model_path,
            dataset_list_txt=dataset_list_txt,
            data_root=data_root,
            zscore_norm=zscore_norm,
        )
    elif model_type == "esmif":
        return ESMIFDataset(
            data_path=data_path,
            score_column=score_column,
            max_seq_length=max_seq_length,
            dataset_list_txt=dataset_list_txt,
            data_root=data_root,
            zscore_norm=zscore_norm,
        )
    elif model_type == "mifst":
        return MIFSTDataset(
            data_path=data_path,
            score_column=score_column,
            max_seq_length=max_seq_length,
            dataset_list_txt=dataset_list_txt,
            data_root=data_root,
            zscore_norm=zscore_norm,
        )
    elif model_type == "proteinmpnn":
        return ProteinMPNNDataset(
            data_path=data_path,
            score_column=score_column,
            max_seq_length=max_seq_length,
            dataset_list_txt=dataset_list_txt,
            data_root=data_root,
            zscore_norm=zscore_norm,
        )
    elif model_type == "proteinmpnn-esm":
        return ProteinMPNNESMDataset(
            data_path=data_path,
            score_column=score_column,
            max_seq_length=max_seq_length,
            esm_model_path=esm_model_path or "facebook/esm2_t33_650M_UR50D",
            dataset_list_txt=dataset_list_txt,
            data_root=data_root,
            zscore_norm=zscore_norm,
        )
    elif model_type == "prosst":
        return ProSSTDataset(
            data_path=data_path,
            score_column=score_column,
            max_seq_length=max_seq_length,
            model_path=model_path,
            prosst_data_path=prosst_data_path,
            dataset_list_txt=dataset_list_txt,
            data_root=data_root,
            zscore_norm=zscore_norm,
        )
    else:
        raise ValueError(f"Invalid dataset type: {model_type}")


def make_train_dataset(args):
    score_column = getattr(args, "score_column", None)
    return _make_dataset(
        model_type=args.model_type,
        data_path=args.train_data_path,
        score_column=score_column,
        max_seq_length=args.max_seq_length,
        model_path=getattr(args, "model_path", None),
        esm_model_path=getattr(args, "esm_model_path", None),
        prosst_data_path=getattr(args, "prosst_data_path", "dataset/data/prosst"),
        dataset_list_txt=getattr(args, "train_list_txt", None),
        data_root=getattr(args, "data_root", "dataset/data"),
        zscore_norm=getattr(args, "train_zscore_norm", False),
    )


def make_valid_dataset(args):
    score_column = getattr(args, "valid_score_column", getattr(args, "score_column", None))
    return _make_dataset(
        model_type=args.model_type,
        data_path=args.valid_data_path,
        score_column=score_column,
        max_seq_length=args.max_seq_length,
        model_path=getattr(args, "model_path", None),
        esm_model_path=getattr(args, "esm_model_path", None),
        prosst_data_path=getattr(args, "prosst_data_path", "dataset/data/prosst"),
        dataset_list_txt=getattr(args, "valid_list_txt", None),
        data_root=getattr(args, "data_root", "dataset/data"),
        zscore_norm=False,
    )


def make_test_dataset(args):
    score_column = getattr(args, "test_score_column", getattr(args, "score_column", None))
    return _make_dataset(
        model_type=args.model_type,
        data_path=args.test_data_path,
        score_column=score_column,
        max_seq_length=args.max_seq_length,
        model_path=getattr(args, "model_path", None),
        esm_model_path=getattr(args, "esm_model_path", None),
        prosst_data_path=getattr(args, "prosst_data_path", "dataset/data/prosst"),
        dataset_list_txt=getattr(args, "test_list_txt", None),
        data_root=getattr(args, "data_root", "dataset/data"),
        zscore_norm=False,
    )
