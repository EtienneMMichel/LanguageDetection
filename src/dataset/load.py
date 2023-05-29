import datasets

DATASET = "wili_2018"

def info():
    ds_builder = datasets.load_dataset_builder(DATASET)
    print(ds_builder.info.description)
    print(ds_builder.info.features)


def execute(selected_lang=None):
    train_ds, test_ds = datasets.load_dataset(
        DATASET,
        split=['train', 'test']
    )
    if not selected_lang is None:
        train_ds = train_ds.filter(lambda elm: elm['label'] in selected_lang)
        test_ds = test_ds.filter(lambda elm: elm['label'] in selected_lang)

    return train_ds, test_ds

if __name__ == "__main__":

    info()

