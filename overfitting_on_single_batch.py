import torch
import torch.nn as nn
# overfit on one batch
from model import SmolLM, SmolLMConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

model = SmolLM(SmolLMConfig())

dataset = load_dataset(
    "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True
)
dataloader = DataLoader(dataset, batch_size=32)

batch = next(iter(dataloader))

batch.shape


tokenizer = GPT2Tokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.vocab_size


def load_cosmopedia_dataset(batch_size=8, seq_length=1024):
    """
    Returns a torch dataloader for the cosmopedia dataset
    """
    try:
        dataset = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            name="cosmopedia-v2",
            split="train",
            streaming=True,
        )

        def encode(examples):
            tokens = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=seq_length + 1,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].squeeze(0).clone().detach()
            input_ids = torch.clamp(input_ids, min=0, max=tokenizer.vocab_size - 1)
            labels = input_ids.clone().detach()
            labels = labels[1:].to(torch.int64)
            input_ids = input_ids[:-1].to(torch.int64)

            return {"input_ids": input_ids, "labels": labels}

        dataset = dataset.map(encode, remove_columns=["text"], batched=False)
        dataset = dataset.with_format("torch")
        dataloader = DataLoader(dataset, batch_size=batch_size)
        return dataloader
    except Exception as e:
        print(e)
        return None

dataloader = load_cosmopedia_dataset(batch_size=8, seq_length=32)
batch = next(iter(dataloader))
batch.keys(), batch["input_ids"].shape, batch["labels"].shape

# calculate loss
loss = nn.CrossEntropyLoss()
logits, loss = model(batch["input_ids"], batch["labels"])
loss
# tensor(10.9244, grad_fn=<NllLossBackward0>)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# overfit on one batch
for i in range(100):
    optimizer.zero_grad()
    _, loss = model(batch["input_ids"], batch["labels"])
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item()}")

# Loss: 10.89421272277832 -ln(vocab_size[49152])
# Loss: 9.869040489196777
# Loss: 8.853536605834961
# Loss: 8.781745910644531
# Loss: 7.623824119567871
# Loss: 6.837253093719482
# Loss: 6.178323268890381
# Loss: 5.542541027069092
# Loss: 5.0422797203063965
# Loss: 4.686131954193115
# Loss: 4.399440765380859
# Loss: 4.189061641693115
# Loss: 4.018494129180908
# Loss: 3.9484009742736816
# Loss: 3.851135015487671
# Loss: 3.7527835369110107
# Loss: 3.681689977645874
# Loss: 3.631518602371216
# Loss: 3.5966415405273438
# Loss: 3.567251443862915
# Loss: 3.536252737045288
# Loss: 3.4979593753814697
# Loss: 3.4603471755981445
# Loss: 3.42283034324646
# Loss: 3.73797607421875
# Loss: 3.422252655029297
# Loss: 3.369056224822998
# Loss: 3.32795786857605
# Loss: 3.2867016792297363
# Loss: 3.469588041305542
# Loss: 3.6112396717071533
# Loss: 3.239042282104492
# Loss: 3.5233707427978516
# Loss: 3.186410427093506
# Loss: 3.147298574447632
# Loss: 3.0759565830230713
# Loss: 3.0572314262390137
# Loss: 3.0441136360168457
# Loss: 3.0136330127716064
# Loss: 2.9667727947235107
# Loss: 2.939312219619751
# Loss: 2.9142584800720215
# Loss: 2.8840432167053223
# Loss: 2.94073224067688
# Loss: 2.942589044570923
# Loss: 2.8911261558532715
# Loss: 2.8405373096466064
# Loss: 2.810102939605713
# Loss: 2.7568352222442627
# Loss: 2.731308698654175
# Loss: 2.6883697509765625
# Loss: 2.66229510307312
# Loss: 2.638798713684082
# Loss: 2.61786150932312
# Loss: 2.5666677951812744
# Loss: 2.522183895111084
# Loss: 2.502363681793213
# Loss: 2.4705874919891357
# Loss: 2.40968918800354
# Loss: 2.367358684539795
# Loss: 2.3259201049804688
# Loss: 2.265883684158325
# Loss: 2.235816240310669
# Loss: 2.192800521850586
# Loss: 2.116086006164551
# Loss: 2.088467597961426
# Loss: 2.0888586044311523
# Loss: 1.9747581481933594
# Loss: 1.9392879009246826
# Loss: 1.9179232120513916
# Loss: 1.964686393737793
# Loss: 1.8549442291259766
# Loss: 1.872979998588562
# Loss: 1.764844536781311
# Loss: 1.7118353843688965
# Loss: 1.6020375490188599
# Loss: 1.58417546749115
# Loss: 1.5007879734039307
# Loss: 1.4107234477996826
# Loss: 1.3546488285064697
# Loss: 1.2871146202087402
# Loss: 1.228489637374878
# Loss: 1.19940185546875
# Loss: 1.1488451957702637
# Loss: 1.1999956369400024
# Loss: 0.9937475323677063
# Loss: 1.0546135902404785
# Loss: 0.9591179490089417
# Loss: 0.9203736186027527
# Loss: 0.8559925556182861
# Loss: 0.811633288860321
# Loss: 0.7453390955924988
# Loss: 0.6871188282966614
# Loss: 0.6159484386444092
# Loss: 0.6068990230560303
# Loss: 0.5419051647186279
# Loss: 0.5068542957305908
# Loss: 0.48658642172813416
# Loss: 0.4673026502132416
# Loss: 0.4020301401615143
