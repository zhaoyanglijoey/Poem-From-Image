# Poem-From-Image
Final project for CMPT419: Generate poem from image

## Download dataset

```bash
python download_image.py
```

## Extract poem & img feature

```
python extract_feature -s poem
python extract_feature -s img
```

Output will be a dictionary of (id, feature) pairs
where feature is a 512 dimension numpy vector.

Output will be saved to `data/poem_features.plk` or 
`data/img_features.plk`

## Unim poem training

```
python vocab_builder.py 
python unim_train.py
```

