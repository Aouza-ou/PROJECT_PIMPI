import torch

ckpt = torch.load("checkpoints/PIMPI_500.pt", map_location="cpu", weights_only=False)

print("Epoch:", ckpt["epoch"])
print("Val loss:", ckpt["val_loss"])
print("Vocab size:", ckpt["vocab_size"])
print("Tokenizer vocab entries:", len(ckpt["tokenizer_vocab"]))

enc = ckpt["encoder_state_dict"]
dec = ckpt["decoder_state_dict"]

print("\nEncoder parameters:", len(enc))
for name, tensor in list(enc.items())[:5]:   # first 5 as a sample
    print(f"  {name}: {tuple(tensor.shape)}")

print("\nDecoder parameters:", len(dec))
for name, tensor in list(dec.items())[:5]:
    print(f"  {name}: {tuple(tensor.shape)}")

total = sum(t.numel() for t in enc.values()) + sum(t.numel() for t in dec.values())
print(f"\nTotal parameters: {total:,} (~{total/1e6:.2f}M)")