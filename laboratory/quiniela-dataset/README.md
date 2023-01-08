# Quiniela Dataset for testing

## How to download and build your dataset:

1. Create your dynenv


```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

2. Process and fix your file

```bash
python 1_*y
python 2_p*y
python 2_1*y
python 2_2*y
python 3_*y
```

You will get and output in datasets/ directory. Check the files because there's some manual fix in the dataset.
