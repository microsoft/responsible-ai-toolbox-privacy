from datasets import Dataset, features


def drop_null_rows(ds: Dataset):
    return ds.filter(lambda x: all(x is not None for x in x.values()), keep_in_memory=True)


def test_null_rows():
    ds = Dataset.from_dict({
            "a": [1, 2, None, 4, 5],
            "b": [1, 2, None, 4, 5],
        }, features=features.Features({
            "a": features.Value("int64"),
            "b": features.Value("int64"),
        })
    )

    ds = drop_null_rows(ds)

    assert len(ds) == 4

