import torch

data = torch.load("heterodata.pt", weights_only=False)
x = data['node'].x

# Replace N with the pattern index you identified as "computer" from the script above
COMPUTER_COLUMN = N  # the one-hot column index, not the pattern index
computer_mask = x[:, COMPUTER_COLUMN] == 1
computer_indices = computer_mask.nonzero().flatten()
print(f"Number of computer nodes: {computer_indices.shape[0]}")

print("\n--- Edge direction audit for computer nodes ---")
for edge_type in data.edge_types:
    src, rel, dst = edge_type
    ei = data[edge_type].edge_index
    if ei.shape[1] == 0:
        continue
    src_is_computer = torch.isin(ei[0], computer_indices).sum().item()
    dst_is_computer = torch.isin(ei[1], computer_indices).sum().item()
    if src_is_computer > 0 or dst_is_computer > 0:
        print(f"{rel:30s} | computer as SOURCE: {src_is_computer:5d} | computer as TARGET: {dst_is_computer:5d}")
