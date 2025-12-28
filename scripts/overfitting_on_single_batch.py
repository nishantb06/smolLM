import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
# overfit on one batch
# from scripts.model import SmolLM, SmolLMConfig
from scripts.model_mlha_moe import SmolLM, SmolLMConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

model = SmolLM(SmolLMConfig())

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

# mlha logs for overfitting
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

# mlha + moe logs for overfitting
# Loss: 10.901752471923828
# Loss: 10.112021446228027
# Loss: 8.672393798828125
# Loss: 8.530308723449707
# Loss: 6.783775329589844
# Loss: 4.726860046386719
# Loss: 3.8616816997528076
# Loss: 2.3563222885131836
# Loss: 1.5700621604919434
# Loss: 1.0685497522354126
# Loss: 0.7385364770889282
# Loss: 0.5166082978248596
# Loss: 0.38537999987602234
# Loss: 0.2897229492664337
# Loss: 0.2165244072675705
# Loss: 0.18605147302150726
# Loss: 0.13685238361358643
# Loss: 0.1810222864151001
# Loss: 0.29188987612724304
# Loss: 0.310602068901062
# Loss: 0.3470343053340912
# Loss: 0.3073788583278656
# Loss: 0.2503785192966461
# Loss: 0.29169338941574097
# Loss: 0.13146960735321045
# Loss: 0.25691238045692444
# Loss: 0.25982576608657837
# Loss: 0.2184397131204605
# Loss: 0.19409218430519104
# Loss: 0.27606579661369324
# Loss: 0.2997124493122101
# Loss: 0.25295108556747437
# Loss: 0.2710171639919281
# Loss: 0.309805691242218
# Loss: 0.2617306709289551
# Loss: 0.3269665241241455
# Loss: 0.20741771161556244
# Loss: 0.29039841890335083
# Loss: 0.1760159283876419
# Loss: 0.22579814493656158
# Loss: 0.2717730402946472
# Loss: 0.22382375597953796
# Loss: 0.32294338941574097
# Loss: 0.21277622878551483
# Loss: 0.2353154718875885
# Loss: 0.2268526256084442
# Loss: 0.27481433749198914
# Loss: 0.19220194220542908
# Loss: 0.1922033578157425
# Loss: 0.17213188111782074
# Loss: 0.10640981793403625
# Loss: 0.1035793125629425
# Loss: 0.1799832135438919
# Loss: 0.13383707404136658
# Loss: 0.11145950853824615
# Loss: 0.17184682190418243
# Loss: 0.13595348596572876
# Loss: 0.1289081871509552
# Loss: 0.21437691152095795
# Loss: 0.130544513463974
# Loss: 0.2765410840511322
# Loss: 0.12140099704265594
# Loss: 0.10958646982908249
# Loss: 0.10141950100660324
# Loss: 0.16040551662445068
# Loss: 0.18532468378543854
# Loss: 0.24288739264011383
# Loss: 0.16458560526371002
# Loss: 0.16964685916900635
# Loss: 0.1258372813463211
# Loss: 0.08830955624580383
# Loss: 0.09478546679019928
# Loss: 0.09516765177249908
# Loss: 0.07238566875457764
# Loss: 0.09564658254384995
# Loss: 0.10524971038103104
# Loss: 0.21952538192272186
# Loss: 0.21090179681777954
# Loss: 0.21952074766159058
# Loss: 0.16278864443302155
# Loss: 0.22013939917087555
# Loss: 0.1562957912683487
# Loss: 0.163138285279274
# Loss: 0.1849743127822876
# Loss: 0.1691092848777771
# Loss: 0.1867380142211914
# Loss: 0.16266855597496033
# Loss: 0.20283354818820953
# Loss: 0.13595914840698242
# Loss: 0.20097628235816956
# Loss: 0.09127158671617508
# Loss: 0.11318465322256088
# Loss: 0.12698234617710114
# Loss: 0.09494362026453018
# Loss: 0.11598867923021317
# Loss: 0.1121702566742897
# Loss: 0.12449034303426743
# Loss: 0.20971213281154633
# Loss: 0.20402181148529053
# Loss: 0.1897549331188202

# 