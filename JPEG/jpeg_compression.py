import math
import cv2 
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt

img = cv2.imread("Rainier.bmp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, c = img.shape

# IMPROVED: Vectorized RGB to YCbCr conversion (much faster)
R = img[:, :, 0].astype(np.float32)
G = img[:, :, 1].astype(np.float32)
B = img[:, :, 2].astype(np.float32)

Y  = np.clip(0.299*R + 0.587*G + 0.114*B, 0, 255).astype(np.uint8)
Cb = np.clip(-0.1687*R - 0.3313*G + 0.5*B + 128, 0, 255).astype(np.uint8)
Cr = np.clip(0.5*R - 0.4187*G - 0.0813*B + 128, 0, 255).astype(np.uint8)

# Ensure dimensions are even
h_even = h - (h % 2)
w_even = w - (w % 2)

Cb_copy = Cb.copy()
Cr_copy = Cr.copy()


for i in range(0, h_even, 2):
    for j in range(0, w_even, 2):
        # Average 2x2 blocks
        avg_cb = (int(Cb[i, j]) + int(Cb[i+1, j]) + 
                  int(Cb[i, j+1]) + int(Cb[i+1, j+1])) / 4.0
        avg_cr = (int(Cr[i, j]) + int(Cr[i+1, j]) + 
                  int(Cr[i, j+1]) + int(Cr[i+1, j+1])) / 4.0
        
        # Assign averaged value to all 4 pixels
        Cb_copy[i:i+2, j:j+2] = np.uint8(avg_cb)
        Cr_copy[i:i+2, j:j+2] = np.uint8(avg_cr)

# Handle edge pixels if dimensions were odd
if h % 2 != 0:
    Cb_copy[-1, :] = Cb[-1, :]
    Cr_copy[-1, :] = Cr[-1, :]
if w % 2 != 0:
    Cb_copy[:, -1] = Cb[:, -1]
    Cr_copy[:, -1] = Cr[:, -1]



# Dividing the Luminance into 8x8 blocks of pixels
blocks_Y = []

for i in range(0, h, 8):
    for j in range(0, w, 8):
        blocks_Y.append(Y[i : i+8, j : j+8])
        

def dct2(block):
    block = block.astype(np.float32) - 128;
    return cv2.dct(block)


dct_blocks_Y = []

for block in blocks_Y:
    if block.shape == (8,8):
        dct_blocks_Y.append(dct2(block));

        
# Dividing the Red Chrominance into 8x8 block of pixels
blocks_R = []

for i in range(0, h, 8):
    for j in range(0, w, 8):
        blocks_R.append(Cr[i : i+8, j : j+8])


dct_blocks_R = []

for block in blocks_R:
    if block.shape == (8,8):
        dct_blocks_R.append(dct2(block));
        


# Dividing the Blue Chrominance into 8x8 block of pixels
blocks_B = []

for i in range(0, h, 8):
    for j in range(0, w, 8):
        blocks_B.append(Cb[i : i+8, j : j+8])
    

dct_blocks_B = []

for block in blocks_B:
    if block.shape == (8,8):
        dct_blocks_B.append(dct2(block));
        
        
# Quantization for Luminance

Q_Luminance = np.array([
    [4, 3, 4, 4, 4, 6, 11,15],
    [3, 3, 3, 4, 5, 8, 14, 19],
    [3, 4, 4, 5, 8, 12, 16, 20],
    [4, 5, 6, 7, 12, 14, 18, 20],
    [6, 6, 9, 11, 14, 17, 21, 23],
    [9, 12, 12, 18, 23, 22, 25, 21],
    [11, 13, 15, 17, 21, 23, 25, 21],
    [13, 12, 12, 13, 16, 19, 21, 21]
], dtype = np.float32)

Q_Chrominance = np.array([
    [4, 4, 6, 10, 21, 21, 21, 21],
    [4, 5, 6, 21, 21, 21, 21, 21],
    [6, 6, 12, 21, 21, 21, 21, 21],
    [10, 14, 21, 21, 21, 21, 21, 21],
    [21, 21, 21, 21, 21, 21, 21, 21],
    [21, 21, 21, 21, 21, 21, 21, 21],
    [21, 21, 21, 21, 21, 21, 21, 21],
    [21, 21, 21, 21, 21, 21, 21, 21]
], dtype = np.float32)


quantized_blocks_Y = []

for dct_block in dct_blocks_Y:
    quantized_block = np.round(dct_block / Q_Luminance)
    quantized_blocks_Y.append(quantized_block)


quantized_blocks_B = []

for dct_block in dct_blocks_B:
    quantized_block = np.round(dct_block / Q_Chrominance)
    quantized_blocks_B.append(quantized_block)    
    
    
quantized_blocks_R = []

for dct_block in dct_blocks_R:
    quantized_block = np.round(dct_block / Q_Chrominance)
    quantized_blocks_R.append(quantized_block)







# Zigzag scan pattern for 8x8 block
def zigzag_pattern():
    """Generate zigzag scan order for 8x8 block"""
    zigzag = []
    for i in range(15):  # 0 to 14 (sum of indices)
        if i % 2 == 0:  # even diagonal - go down
            for j in range(min(i+1, 8)):
                row = i - j
                col = j
                if row < 8 and col < 8:
                    zigzag.append((row, col))
        else:  # odd diagonal - go up
            for j in range(min(i+1, 8)):
                row = j
                col = i - j
                if row < 8 and col < 8:
                    zigzag.append((row, col))
    return zigzag


def zigzag_scan(block):
    """Convert 8x8 block to 1D array using zigzag pattern"""
    pattern = zigzag_pattern()
    return [block[i, j] for i, j in pattern]


# Apply zigzag scanning to all quantized blocks
zigzag_Y = [zigzag_scan(block) for block in quantized_blocks_Y]
zigzag_B = [zigzag_scan(block) for block in quantized_blocks_B]
zigzag_R = [zigzag_scan(block) for block in quantized_blocks_R]


# Run-Length Encoding (RLE)
def run_length_encode(zigzag_sequence):
    """
    Encode zigzag sequence using RLE
    Returns list of (run_length, value) tuples
    """
    rle = []
    zero_count = 0
    
    for value in zigzag_sequence:
        if value == 0:
            zero_count += 1
        else:
            # Store (number of zeros before this value, value)
            rle.append((zero_count, int(value)))
            zero_count = 0
    
    # End of block marker (0, 0) if there are trailing zeros
    if zero_count > 0:
        rle.append((0, 0))  # EOB marker
    
    return rle


# Apply RLE to all zigzag sequences
rle_Y = [run_length_encode(seq) for seq in zigzag_Y]
rle_B = [run_length_encode(seq) for seq in zigzag_B]
rle_R = [run_length_encode(seq) for seq in zigzag_R]


# Huffman Coding
from collections import Counter
import heapq


class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(frequencies):
    """Build Huffman tree from frequency dictionary"""
    if not frequencies:
        return None
    
    heap = [HuffmanNode(symbol=sym, freq=freq) for sym, freq in frequencies.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, parent)
    
    return heap[0]


def generate_huffman_codes(root, code="", codes=None):
    """Generate Huffman codes from tree"""
    if codes is None:
        codes = {}
    
    if root is None:
        return codes
    
    if root.symbol is not None:  # Leaf node
        codes[root.symbol] = code if code else "0"
        return codes
    
    generate_huffman_codes(root.left, code + "0", codes)
    generate_huffman_codes(root.right, code + "1", codes)
    
    return codes


def huffman_encode(rle_data):
    """
    Encode RLE data using Huffman coding
    Returns: encoded bitstring and huffman codes dictionary
    """
    # Flatten all RLE tuples to get symbols
    all_symbols = []
    for block_rle in rle_data:
        for run, value in block_rle:
            all_symbols.append((run, value))
    
    # Calculate frequencies
    frequencies = Counter(all_symbols)
    
    # Build Huffman tree and generate codes
    tree = build_huffman_tree(frequencies)
    huffman_codes = generate_huffman_codes(tree)
    
    # Encode the data
    encoded_bits = []
    for block_rle in rle_data:
        for symbol in block_rle:
            encoded_bits.append(huffman_codes[symbol])
    
    return ''.join(encoded_bits), huffman_codes


# Apply Huffman encoding
encoded_Y, huffman_codes_Y = huffman_encode(rle_Y)
encoded_B, huffman_codes_B = huffman_encode(rle_B)
encoded_R, huffman_codes_R = huffman_encode(rle_R)



# ============================================================================
# DECOMPRESSION / DECODING PROCESS
# ============================================================================

def huffman_decode(encoded_bits, huffman_codes, num_blocks):
    """
    Decode Huffman encoded bitstring back to RLE data
    """
    # Create reverse mapping: code -> symbol
    reverse_codes = {code: symbol for symbol, code in huffman_codes.items()}
    
    decoded_rle = []
    current_code = ""
    block_rle = []
    
    for bit in encoded_bits:
        current_code += bit
        
        if current_code in reverse_codes:
            symbol = reverse_codes[current_code]
            block_rle.append(symbol)
            
            # Check for End of Block marker
            if symbol == (0, 0):
                decoded_rle.append(block_rle)
                block_rle = []
                if len(decoded_rle) == num_blocks:
                    break
            
            current_code = ""
    
    # Handle last block if it doesn't have EOB marker
    if block_rle and len(decoded_rle) < num_blocks:
        decoded_rle.append(block_rle)
    
    return decoded_rle


def run_length_decode(rle_sequence):
    """
    Decode RLE sequence back to zigzag sequence
    """
    zigzag = []
    
    for run, value in rle_sequence:
        # Add run_length zeros
        zigzag.extend([0] * run)
        
        # Add the value (unless it's EOB marker)
        if (run, value) != (0, 0):
            zigzag.append(value)
    
    # Pad to 64 elements if needed
    while len(zigzag) < 64:
        zigzag.append(0)
    
    return zigzag[:64]  # Ensure exactly 64 elements


def inverse_zigzag_scan(zigzag_sequence):
    """Convert 1D zigzag sequence back to 8x8 block"""
    block = np.zeros((8, 8), dtype=np.float32)
    pattern = zigzag_pattern()
    
    for idx, (i, j) in enumerate(pattern):
        if idx < len(zigzag_sequence):
            block[i, j] = zigzag_sequence[idx]
    
    return block


def idct2(block):
    """Inverse DCT"""
    block = cv2.idct(block)
    return block + 128


# Decode Huffman encoded data
decoded_rle_Y = huffman_decode(encoded_Y, huffman_codes_Y, len(quantized_blocks_Y))
decoded_rle_B = huffman_decode(encoded_B, huffman_codes_B, len(quantized_blocks_B))
decoded_rle_R = huffman_decode(encoded_R, huffman_codes_R, len(quantized_blocks_R))


# Decode RLE to zigzag sequences
decoded_zigzag_Y = [run_length_decode(rle) for rle in decoded_rle_Y]
decoded_zigzag_B = [run_length_decode(rle) for rle in decoded_rle_B]
decoded_zigzag_R = [run_length_decode(rle) for rle in decoded_rle_R]


# Convert zigzag back to 8x8 blocks
decoded_quantized_Y = [inverse_zigzag_scan(zz) for zz in decoded_zigzag_Y]
decoded_quantized_B = [inverse_zigzag_scan(zz) for zz in decoded_zigzag_B]
decoded_quantized_R = [inverse_zigzag_scan(zz) for zz in decoded_zigzag_R]


# Dequantize (multiply by quantization tables)
dequantized_Y = [block * Q_Luminance for block in decoded_quantized_Y]
dequantized_B = [block * Q_Chrominance for block in decoded_quantized_B]
dequantized_R = [block * Q_Chrominance for block in decoded_quantized_R]


# Apply Inverse DCT
idct_blocks_Y = [idct2(block) for block in dequantized_Y]
idct_blocks_B = [idct2(block) for block in dequantized_B]
idct_blocks_R = [idct2(block) for block in dequantized_R]


# Reconstruct full Y, Cb, Cr channels from blocks
def reconstruct_channel(blocks, height, width):
    """Reconstruct full channel from 8x8 blocks"""
    channel = np.zeros((height, width), dtype=np.float32)
    block_idx = 0
    
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            if block_idx < len(blocks):
                block = blocks[block_idx]
                h_end = min(i + 8, height)
                w_end = min(j + 8, width)
                channel[i:h_end, j:w_end] = block[:h_end-i, :w_end-j]
                block_idx += 1
    
    return np.clip(channel, 0, 255).astype(np.uint8)


Y_reconstructed = reconstruct_channel(idct_blocks_Y, h, w)
Cb_reconstructed = reconstruct_channel(idct_blocks_B, h, w)
Cr_reconstructed = reconstruct_channel(idct_blocks_R, h, w)


# Convert YCbCr back to RGB
Y_f = Y_reconstructed.astype(np.float32)
Cb_f = Cb_reconstructed.astype(np.float32)
Cr_f = Cr_reconstructed.astype(np.float32)

R_reconstructed = np.clip(Y_f + 1.402 * (Cr_f - 128), 0, 255).astype(np.uint8)
G_reconstructed = np.clip(Y_f - 0.34414 * (Cb_f - 128) - 0.71414 * (Cr_f - 128), 0, 255).astype(np.uint8)
B_reconstructed = np.clip(Y_f + 1.772 * (Cb_f - 128), 0, 255).astype(np.uint8)

# Create final reconstructed image
img_reconstructed = np.stack([R_reconstructed, G_reconstructed, B_reconstructed], axis=2)


# ============================================================================
# VISUALIZATION AND COMPARISON
# ============================================================================

# Calculate PSNR
mse = np.mean((img.astype(np.float32) - img_reconstructed.astype(np.float32)) ** 2)
if mse == 0:
    psnr = float('inf')
else:
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))

# Display original vs compressed image
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(img_reconstructed)
axes[1].set_title(f"Compressed Image")
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Save compressed image
cv2.imwrite("Rainier_compressed.png", cv2.cvtColor(img_reconstructed, cv2.COLOR_RGB2BGR))































