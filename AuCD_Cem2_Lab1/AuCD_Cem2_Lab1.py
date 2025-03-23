import numpy as np
import queue

class Node():
    def __init__(self, symbol=None, counter=None, left=None, right=None, parent=None):
        self.symbol = symbol
        self.counter = counter
        self.left = left
        self.right = right
        self.parent = parent

    def __lt__(self, other):
        return self.counter < other.counter

def count_symb(data: bytes) -> np.ndarray:
    counter = np.zeros(256, dtype=int)
    for byte in data:
        counter[byte] += 1
    return counter

def huffman_compress(data: bytes) -> bytes:
    C = count_symb(data)
    list_of_leafs = []
    Q = queue.PriorityQueue()

    for i in range(256):
        if C[i] != 0:
            leaf = Node(symbol=i, counter=C[i])
            list_of_leafs.append(leaf)
            Q.put(leaf)

    while Q.qsize() >= 2:
        left_node = Q.get()
        right_node = Q.get()
        parent_node = Node(left=left_node, right=right_node)
        left_node.parent = parent_node
        right_node.parent = parent_node
        parent_node.counter = left_node.counter + right_node.counter
        Q.put(parent_node)

    codes = {}
    for leaf in list_of_leafs:
        node = leaf
        code = ""
        while node.parent is not None:
            if node.parent.left == node:
                code = "0" + code
            else:
                code = "1" + code
            node = node.parent
        codes[leaf.symbol] = code

    coded_message = ""
    for byte in data:
        coded_message += codes[byte]

    padding = 8 - len(coded_message) % 8
    coded_message += '0' * padding
    coded_message = f"{padding:08b}" + coded_message
    bytes_string = bytearray()
    for i in range(0, len(coded_message), 8):
        byte = coded_message[i:i+8]
        bytes_string.append(int(byte, 2))
    
    return bytes(bytes_string), codes

def huffman_decompress(compressed_data: bytes, huffman_codes: dict) -> bytes:
    padding = compressed_data[0]
    coded_message = ""
    for byte in compressed_data[1:]:
        coded_message += f"{byte:08b}"

    if padding > 0:
        coded_message = coded_message[:-padding]    


    reverse_codes = {v: k for k, v in huffman_codes.items()}
    current_code = ""
    decoded_data = bytearray()

    for bit in coded_message:
        current_code += bit
        if current_code in reverse_codes:
            decoded_data.append(reverse_codes[current_code])
            current_code = ""

    return bytes(decoded_data)

def read_huffman_codes(codes_file):
    huffman_codes = {}
    with open(codes_file, 'r') as f:
        for line in f:
            symbol, code = line.strip().split(':')
            huffman_codes[int(symbol)] = code
    return huffman_codes

def write_huffman_codes(huffman_codes, file_path):
    with open(file_path, 'w') as code_file:
        for symbol, code in huffman_codes.items():
            code_file.write(f"{symbol}:{code}\n")

def process_file_nontext_1(file_path, output_compressed, output_decompressed):
    with open(file_path, "rb") as f:
        data = f.read()

    compressed_bytes, huffman_codes = huffman_compress(data)

    with open(output_compressed, "wb") as file:
        file.write(compressed_bytes)

    with open(output_compressed + '_codes', 'w') as code_file:
        for symbol, code in huffman_codes.items():
            code_file.write(f"{symbol}:{code}\n")

    with open(output_compressed, "rb") as f:
            compressed_data = f.read()

    huffman_codes = read_huffman_codes(output_compressed + '_codes')

    decompressed_data = huffman_decompress(compressed_data, huffman_codes)

    with open(output_decompressed, "wb") as file:
        file.write(decompressed_data)

    if data == decompressed_data:
        print("Данные совпадают!")
    else:
        print("Данные не совпадают...")

############################################################################################################

def rle_encode_bits(data: bytes, m: int) -> bytes:
    if not data:
        return b''

    compressed_data = bytearray()
    unique = bytearray()
    count = 0
    flag = False
    total_bits = len(data) * 8
    prev_bits = None

    control = (m + 7) // 8

    def get_bits(offset, length):
        bits = 0
        for i in range(length):
            byte_index = (offset + i) // 8
            bit_index = (offset + i) % 8
            if byte_index < len(data):
                bits <<= 1
                bits |= (data[byte_index] >> (7 - bit_index)) & 1
        return bits

    def M_more_then_8(prev_bits: int, array: bytearray):
        if m > 8:
            prev_bits_bin = bin(prev_bits)[2:].zfill(8 * control)
            for i in range(0, len(prev_bits_bin), 8):
                prev_bits_part = prev_bits_bin[i:i + 8]
                array.append(int(prev_bits_part, 2))

            remaining_bits = prev_bits_bin[len(prev_bits_bin) // 8 * 8:]
            if remaining_bits:
                array.append(int(remaining_bits, 2))
        else:
            array.append(prev_bits)

    for i in range(0, total_bits, m):
        next_bits = get_bits(i, m)

        if prev_bits is None:
            prev_bits = next_bits
            count = 1
            continue

        if next_bits == prev_bits:
            count += 1
            if count >= 128:
                compressed_data.append(127)
                M_more_then_8(prev_bits, compressed_data)
                count = 1
        else:
            if count > 0:
                compressed_data.append(count)
                M_more_then_8(prev_bits, compressed_data)
            prev_bits = next_bits
            count = 1

    if count > 0:
        compressed_data.append(count)
        M_more_then_8(prev_bits, compressed_data)

    return bytes(compressed_data)

def rle_decode_bits(compressed_data: bytes, m: int) -> bytes:
    if not compressed_data:
        return b''

    control = (m + 7) // 8

    decoded_data = bytearray()
    byte_dec = ''
    i = 0

    while i < len(compressed_data):
        count = int(compressed_data[i])
        i += 1
        byte_beta = ''

        if count < 128:
            for j in range(control):
                if i < len(compressed_data):
                    prototype = bin(compressed_data[i])[2:].zfill(m)
                    byte_beta += prototype
                    i += 1
            for _ in range(count):
                byte_dec += byte_beta
            
        else:
            count -= 128
            while count > 0:
                for j in range(control):
                    if i < len(compressed_data):
                        prototype = bin(compressed_data[i])[2:].zfill(m)
                        byte_dec += prototype
                        i += 1
                count -= 1

        while len(byte_dec) >= 8:
            decimal_value = int(byte_dec[:8], 2)
            decoded_data.append(decimal_value)
            byte_dec = byte_dec[8:]

    return bytes(decoded_data)

def rle_encode(data,M):
    if not data:
        return b""
    encoded = bytearray()
    block_size =M//8
    count = 1
    repeat_flag=0
    prev_char = data[:block_size ]
    j=0
    data += b'\x00' * block_size 
    i=block_size 
    while (i<len(data)):
        char=data[i:i+block_size ]
        if (char==prev_char and count<127):
            count+=1
            i+=block_size 
            repeat_flag=1
        else:
            if (repeat_flag==1):
                encoded.append(count)
                encoded.extend(prev_char)
                count=1
                repeat_flag=0
                prev_char=char
                i+=block_size 
            else:
                j=0
                dop_str=bytearray()
                while(char!=prev_char and j<127 and i+j*block_size <len(data)):
                    j+=1
                    dop_str.extend(prev_char)
                    prev_char=char
                    char=data[i+j*block_size :i+j*block_size +block_size ] if i+j*block_size <len(data) else b''
                encoded.append(j|0x80)
                encoded.extend(dop_str)
                i+=j*block_size 
    return bytes(encoded)

def rle_decode(data,M):
    if not data:
        return b""
    decoded = bytearray()
    i = 0
    block_size =M//8
    while i < len(data):
        count = (data[i])
        if (count & 0x80):
            length = count& 0x7F
            decoded.extend(data[i+1: i+1+length*block_size ])
            i+=1+length*block_size 
        else:
            char = data[i + 1:i+1+block_size]
            decoded.extend(char * count)
            i += 1+block_size 
    return bytes(decoded)


def process_file_nontext_2(file_path, output_compressed, output_decompressed):
    bit=8
    with open(file_path, "rb") as f:
        data = f.read()

    compressed_bytes= rle_encode(data,bit)
    # compressed_bytes= rle_encode_bits(data,bit)
    print("Compressed")
    with open(output_compressed, "wb") as file:
        file.write(compressed_bytes)

    # decompressed_text = rle_decode(compressed_bytes, bit)
    decompressed_text = rle_decode_bits(compressed_bytes, bit)
    print("DeCompressed")
    with open(output_decompressed, "wb") as file:
        file.write(decompressed_text)

    if data==decompressed_text:
        print("Тексты совпадают!")
    else:
        print("Тексты не совпадают")

############################################################################################################

def process_file_nontext_3(file_path, output_compressed, output_decompressed, block_size=5000):
    with open(file_path, "rb") as f:
        data = f.read()

    blocks = [data[i:i + block_size] for i in range(0, len(data), block_size)]
    compressed_blocks = []

    for block in blocks:
        bwt_result, bwt_index = BWT(block)
        print('BWT complete for block')

        rle_result=rle_encode_bits(bwt_result, 8)
        # rle_result=rle_encode(bwt_result, 8)
        print('RLE complete for block')

        compressed_blocks.append((rle_result, bwt_index))

    with open(output_compressed, "wb") as file:
        for compressed_bytes, bwt_index in compressed_blocks:
            file.write(compressed_bytes)

    decompressed_bwt = bytearray()
    with open(output_compressed, "rb") as file:
        compressed_data = file.read()

    for i, (compressed_bytes, bwt_index) in enumerate(compressed_blocks):
        decompressed_rle = rle_decode_bits(compressed_bytes, 8)
        # decompressed_rle = rle_decode(compressed_bytes, 8)
        print('RLE decompression complete for block')

        block_decompressed_bwt = better_iBWT(decompressed_rle, bwt_index)
        decompressed_bwt += block_decompressed_bwt
        print('better_iBWT complete for block')

    with open(output_decompressed, "wb") as file:
        file.write(decompressed_bwt)

    if data == decompressed_bwt:
        print("Тексты совпадают!")
    else:
        print("Тексты не совпадают")

############################################################################################################

def LZ77(S: bytes) -> list:
    buffer_size = 1024
    string_size = 128
    coding_list = []
    i = 0
    
    while i < len(S):
        buffer = S[max(0, i - buffer_size): i]
        new_buffer_size = len(buffer)
        shift = -1
        
        for j in range(string_size, -1, -1):
            subS = S[i: min(i + j, len(S))]
            shift = buffer.find(subS)
            if shift != -1:
                break
        
        if (i + len(subS) >= len(S)):
            coding_list.append((new_buffer_size - shift, len(subS), b''))
        else:
            coding_list.append((new_buffer_size - shift, len(subS), S[i + len(subS):i + len(subS) + 1]))
        
        i += len(subS) + 1
    
    return coding_list

def iLZ77(compressed_message: list) -> bytes:
    S = bytearray()
    
    for t in compressed_message:
        shift, length, symbol = t
        N = len(S)
        S.extend(S[N - shift:N - shift + length])
        S.extend(symbol)
    
    return bytes(S)

def process_file_nontext_6(file_path, output_compressed, output_decompressed):
    with open(file_path, "rb") as f:
        data = f.read()

    compressed_data = LZ77(data)

    with open(output_compressed, "wb") as file:
        for offset, length, next_char in compressed_data:
            packed_data = (offset.to_bytes(2, byteorder='big')) + bytes([length]) + next_char
            file.write(packed_data)

    with open(output_compressed, "rb") as file:
        compressed_message = []
        while True:
            packed_data = file.read(4)
            if not packed_data:
                break
            offset = int.from_bytes(packed_data[0:2], byteorder='big')
            length = packed_data[2]
            next_char = packed_data[3:4]
            compressed_message.append((offset, length, next_char))

    decompressed_data = iLZ77(compressed_message)

    with open(output_decompressed, "wb") as file:
        file.write(decompressed_data)

    if data == decompressed_data:
        print("Тексты совпадают!")
    else:
        print("Тексты не совпадают")

############################################################################################################

def process_file_nontext_7(file_path, output_compressed, output_decompressed):
    with open(file_path, "rb") as f:
        data = f.read()

    compressed_data = LZ77(data)

    with open(output_compressed, "wb") as file:
        for offset, length, next_char in compressed_data:
            packed_data = (offset.to_bytes(2, byteorder='big')) + bytes([length]) + next_char
            file.write(packed_data)

    with open(output_compressed, "rb") as f:
        data_H = f.read()

    compressed_bytes, huffman_codes = huffman_compress(data_H)

    with open(output_compressed, "wb") as file:
        file.write(compressed_bytes)

    with open(output_compressed + '_codes', 'w') as code_file:
        for symbol, code in huffman_codes.items():
            code_file.write(f"{symbol}:{code}\n")

    with open(output_compressed, "rb") as f:
            compressed_data = f.read()

    huffman_codes = read_huffman_codes(output_compressed + '_codes')

    decompressed_data = huffman_decompress(compressed_data, huffman_codes)

    with open(output_decompressed, "wb") as file:
        file.write(decompressed_data)

    decompressed_data_final = iLZ77(decompressed_data)

    with open(output_decompressed, "wb") as file:
        file.write(decompressed_data_final)

############################################################################################################

def counting_sort(last_column):
    count = [0] * 256
    for char in last_column:
        count[char] += 1

    for i in range(1, 256):
        count[i] += count[i - 1]

    P_inverse = [0] * len(last_column)
    for i in range(len(last_column) - 1, -1, -1):
        char = last_column[i]
        count[char] -= 1
        P_inverse[count[char]] = i

    return P_inverse

def build_suffix_array(s):
    n = len(s)
    suffixes = [(s[i:], i) for i in range(n)]
    suffixes.sort()
    suffix_array = [idx for _, idx in suffixes]
    return suffix_array

def bwt_encode(data):
    if not data:
        return b""
    data += b'\x00'
    n = len(data)
    sa = build_suffix_array(data)
    last_column = bytearray(data[(sa[i] - 1) % n] for i in range(n))
    return bytes(last_column)

def bwt_decode(encoded_data):
    if not encoded_data:
        return b""
    S_index = encoded_data.find(b'\x00')
    if S_index == -1:
        print("Ошибка: не найдена строка, заканчивающаяся на \\x00")
        return b""
    last_column = list(encoded_data)
    P_inverse = counting_sort(last_column)
    N = len(last_column)
    S = bytearray()
    j = S_index
    for _ in range(N):
        j = P_inverse[j]
        S.append(last_column[j])
    return bytes(S).rstrip(b'\x00')


def BWT(S: bytes) -> tuple[bytes, int]:  
    N = len(S)
    BWM = [S[i:] + S[0:i] for i in range(N)] 
    BWM.sort()
    last_column_BWM = bytearray() 
    for i in range(N):
        last_column_BWM.append(BWM[i][-1])  
    S_index = BWM.index(S)
    return bytes(last_column_BWM), S_index

def BWT_blocks(S: bytes, block_size: int) -> tuple[list[bytes], list[int]]:
    blocks = [S[i:i + block_size] for i in range(0, len(S), block_size)]
    bwt_strings = []
    indices = []

    for block in blocks:
        if len(block) < block_size:
            block += b'\0' * (block_size - len(block))
        bwt_string, index = BWT(block)
        bwt_strings.append(bwt_string)
        indices.append(index)

    return bwt_strings, indices

def Merge(S: list[bytes]) -> bytes:
    result = bytearray()
    for i in S:
        result.extend(i)
    return bytes(result)

def iBWT(last_column_BWM: bytes, S_index: int) -> bytes:
    N = len(last_column_BWM)
    BWM = [bytearray() for _ in range(N)]

    for _ in range(N):
        for j in range(N):
            BWM[j] = bytes([last_column_BWM[j]]) + BWM[j]
        BWM.sort()

    S = BWM[S_index]
    return bytes(S)

def better_iBWT(last_column_BWM: bytes, S_index: int) -> bytes:
    N = len(last_column_BWM)
    P_inverse = counting_sort_arg(last_column_BWM)
    S = bytearray()
    j = S_index
    for _ in range(N):
        j = P_inverse[j]
        S.append(last_column_BWM[j])
    return bytes(S)

def counting_sort_arg(S: bytes) -> list[int]:
    N = len(S)
    M = 256
    T = [0 for _ in range(M)]
    T_sub = [0 for _ in range(M)]
    
    for s in S:
        T[s] += 1
    
    for j in range(1, M):
        T_sub[j] = T_sub[j - 1] + T[j - 1]

    P = [-1 for _ in range(N)]
    P_inverse = [-1 for _ in range(N)]
    
    for i in range(N):
        P_inverse[T_sub[S[i]]] = i
        P[i] = T_sub[S[i]]
        T_sub[S[i]] += 1
    
    return P_inverse

def MTF(S: bytes) -> tuple[list[int], list[bytes]]:
    T = [bytes([i]) for i in range(256)]  
    L = []
    for s in S:
        i = T.index(bytes([s])) 
        L.append(i)
        T = [T[i]] + T[:i] + T[i+1:]
    return L, T

def iMTF(S: list[int]) -> bytes:
    T = [bytes([i]) for i in range(256)]
    S_new = bytearray()
    
    for s in S:
        i = s
        S_new.extend(T[i])
        T = [T[i]] + T[:i] + T[i+1:]

    return bytes(S_new)

def process_file_nontext_4(file_path, output_compressed, output_decompressed, block_size=5000):
    with open(file_path, "rb") as f:
        data = f.read()

    blocks = [data[i:i + block_size] for i in range(0, len(data), block_size)]
    huffman_codes_list = []
    compressed_blocks = []

    for block in blocks:
        bwt_result, bwt_index = BWT(block)
        print('BWT complete for block')

        mtf_result, block_huffman_codes = MTF(bwt_result)
        print('MTF complete for block')

        compressed_bytes, _ = huffman_compress(mtf_result)
        print('Huffman compression complete for block')

        compressed_blocks.append((compressed_bytes, bwt_index))
        huffman_codes_list.append(_)

    with open(output_compressed, "wb") as file:
        for compressed_bytes, _ in compressed_blocks:
            file.write(compressed_bytes)

    with open(output_compressed + '_codes', 'w') as code_file:
        for block_huffman_codes in huffman_codes_list:
            for symbol, code in block_huffman_codes.items():
                code_file.write(f"{symbol}:{code}\n")

    decompressed_bwt = bytearray()
    with open(output_compressed, "rb") as file:
        compressed_data = file.read()

    for (compressed_bytes, bwt_index), block_huffman_codes in zip(compressed_blocks, huffman_codes_list):
        decompressed_mtf = huffman_decompress(compressed_bytes, block_huffman_codes)
        print('Huffman decompression complete for block')

        original_indices = iMTF(decompressed_mtf)
        print('iMTF complete for block')

        block_decompressed_bwt = better_iBWT(original_indices, bwt_index)
        decompressed_bwt.extend(block_decompressed_bwt)
        print('better_iBWT complete for block')

    with open(output_decompressed, "wb") as file:
        file.write(decompressed_bwt)

    if data == decompressed_bwt:
        print("Тексты совпадают!")
    else:
        print("Тексты не совпадают")

############################################################################################################

def process_file_nontext_5(file_path, output_compressed, output_decompressed, block_size=5000):
    with open(file_path, "rb") as f:
        data = f.read()

    blocks = [data[i:i + block_size] for i in range(0, len(data), block_size)]
    huffman_codes_list = []
    compressed_blocks = []

    for block in blocks:
        # bwt_result, bwt_index = BWT(block)
        bwt_result = bwt_encode(block)
        print('BWT complete for block')

        mtf_result, block_huffman_codes = MTF(bwt_result)
        print('MTF complete for block')

        # rle_result=rle_encode_bits(mtf_result, 8)
        rle_result=rle_encode(mtf_result, 8)
        print('RLE complete for block')

        compressed_bytes, _ = huffman_compress(rle_result)
        print('Huffman compression complete for block')

        bwt_index=b'1'
        compressed_blocks.append((compressed_bytes, bwt_index))
        huffman_codes_list.append(_)

    with open(output_compressed, "wb") as file:
        for compressed_bytes, _ in compressed_blocks:
            file.write(compressed_bytes)

    with open(output_compressed + '_codes', 'w') as code_file:
        for block_huffman_codes in huffman_codes_list:
            for symbol, code in block_huffman_codes.items():
                code_file.write(f"{symbol}:{code}\n")

    decompressed_bwt = bytearray()
    with open(output_compressed, "rb") as file:
        compressed_data = file.read()

    for (compressed_bytes, bwt_index), block_huffman_codes in zip(compressed_blocks, huffman_codes_list):
        decompressed_rle = huffman_decompress(compressed_bytes, block_huffman_codes)
        print('Huffman decompression complete for block')

        decompressed_mtf=rle_decode_bits(decompressed_rle, 8)
        # decompressed_mtf=rle_decode(decompressed_rle, 8)
        print('RLE decompression complete for block')

        original_indices = iMTF(decompressed_mtf)
        print('iMTF complete for block')

        # block_decompressed_bwt = better_iBWT(original_indices, bwt_index)
        block_decompressed_bwt = bwt_decode(original_indices)
        decompressed_bwt.extend(block_decompressed_bwt)
        print('better_iBWT complete for block')

    with open(output_decompressed, "wb") as file:
        file.write(decompressed_bwt)

    if data == decompressed_bwt:
        print("Тексты совпадают!")
    else:
        print("Тексты не совпадают")

############################################################################################################

def LZ78(data: bytes) -> list:
    coding_list = []
    slovar = {}
    code = 1
    combination = b''

    for char in data:
        combination += bytes([char])
        if combination not in slovar:
            if len(combination) == 1:
                coding_list.append((0, bytes([char])))
            else:
                prev_combination = combination[:-1]
                index = slovar[prev_combination]
                coding_list.append((index, bytes([char])))
            slovar[combination] = code
            code += 1
            combination = b''

    if combination:
        coding_list.append((0, combination))

    return coding_list

def iLZ78(encoded_data: bytes) -> bytes:
    dict_of_codes = {0: b''}
    decoded_data = bytearray()
    i = 0

    while i < len(encoded_data):
        if i + 3 <= len(encoded_data):
            index = int.from_bytes(encoded_data[i:i + 3], byteorder='big')
            i += 3
        else:
            break

        if i < len(encoded_data):
            next_char = encoded_data[i:i + 1]
            i += 1
        else:
            break

        if index in dict_of_codes:
            new_string = dict_of_codes[index] + next_char
            decoded_data.extend(new_string)
            dict_of_codes[len(dict_of_codes)] = new_string
        else:
            decoded_data.extend(next_char)
            dict_of_codes[len(dict_of_codes)] = next_char

    return bytes(decoded_data)

def process_file_nontext_8(file_path, output_compressed, output_decompressed):
    with open(file_path, "rb") as f:
        data = f.read()

    compressed_data = LZ78(data)

    with open(output_compressed, "wb") as file:
        for index, next_char in compressed_data:
            packed_data = (index.to_bytes(3, byteorder='big')) + next_char
            file.write(packed_data)

    print("Compress complete")

    with open(output_compressed, "rb") as file:
        compressed_message = bytearray()
        while True:
            packed_data = file.read(4)
            if not packed_data:
                break
            index = int.from_bytes(packed_data[0:3], byteorder='big')
            next_char = packed_data[3:4]
            compressed_message.extend((index.to_bytes(3, byteorder='big') + next_char))

    decompressed_data = iLZ78(compressed_message)

    with open(output_decompressed, "wb") as file:
        file.write(decompressed_data)

    if data == decompressed_data:
        print("Данные совпали")
    else:
        print("Данные не совпали")

############################################################################################################

def process_file_nontext_7(file_path, output_compressed, output_decompressed):
    with open(file_path, "rb") as f:
        data = f.read()

    compressed_data = LZ78(data)

    with open(output_compressed, "wb") as file:
        for index, next_char in compressed_data:
            packed_data = (index.to_bytes(3, byteorder='big')) + next_char
            file.write(packed_data)

    print("Compress complete")

    with open(output_compressed, "rb") as f:
        data_H = f.read()

    compressed_bytes, huffman_codes = huffman_compress(data_H)

    with open(output_compressed, "wb") as file:
        file.write(compressed_bytes)

    with open(output_compressed + '_codes', 'w') as code_file:
        for symbol, code in huffman_codes.items():
            code_file.write(f"{symbol}:{code}\n")

    with open(output_compressed, "rb") as f:
            compressed_data = f.read()

    huffman_codes = read_huffman_codes(output_compressed + '_codes')

    decompressed_data = huffman_decompress(compressed_data, huffman_codes)

    with open(output_decompressed, "wb") as file:
        file.write(decompressed_data)

    decompressed_data_final = iLZ78(decompressed_data)

    with open(output_decompressed, "wb") as file:
        file.write(decompressed_data_final)

###########################################################################################################3

if __name__ == "__main__":
    compressor = 5
    choice = 1
    if compressor==1:
        if choice == 1:
            process_file_nontext_1("Кафка Франц.txt", "Кафка Франц_c_HA.txt", "Кафка Франц_d_HA.txt")
        elif choice == 2:
            process_file_nontext_1("enwik7", "enwik7_c_HA", "enwik7_d_HA")
        elif choice == 3:
            process_file_nontext_1("idea64.exe", "idea64_c_HA", "idea64_d_HA")
        elif choice == 4:
            process_file_nontext_1("BlackAndWhite.raw", "BlackAndWhite_c_HA.raw", "BlackAndWhite_d_HA.raw")
        elif choice == 5:
            process_file_nontext_1("Gray.raw", "Gray_c_HA.raw", "Gray_d_HA.raw")
        elif choice == 6:
            process_file_nontext_1("Color.raw", "Color_c_HA.raw", "Color_d_HA.raw")
        elif choice == 7:
            process_file_nontext_2("KS_baw.raw", "KS_baw_c_HA.raw", "KS_baw_d_HA.raw")
        elif choice == 8:
            process_file_nontext_2("KS_gray.raw", "KS_gray_c_HA.raw", "KS_gray_d_HA.raw")
        elif choice == 9:
            process_file_nontext_2("KS_color.raw", "KS_color_c_HA.raw", "KS_color_d_HA.raw")
    elif compressor==2:
        if choice == 1:
            process_file_nontext_2("Кафка Франц.txt", "Кафка Франц_c_RLE.txt", "Кафка Франц_d_RLE.txt")
        elif choice == 2:
            process_file_nontext_2("enwik7", "enwik7_c_RLE", "enwik7_d_RLE")
        elif choice == 3:
            process_file_nontext_2("idea64.exe", "idea64_c_RLE", "idea64_d_RLE.exe")
        elif choice == 4:
            process_file_nontext_2("BlackAndWhite.raw", "BlackAndWhite_c_RLE.raw", "BlackAndWhite_d_RLE.raw")
        elif choice == 5:
            process_file_nontext_2("Gray.raw", "Gray_c_RLE.raw", "Gray_d_RLE.raw")
        elif choice == 6:
            process_file_nontext_2("Color.raw", "Color_c_RLE.raw", "Color_d_RLE.raw")
    elif compressor==3:
        if choice == 1:
            process_file_nontext_3("Кафка Франц.txt", "Кафка Франц_c_BWT+RLE.txt", "Кафка Франц_d_BWT+RLE.txt")
        elif choice == 2:
            process_file_nontext_3("enwik7", "enwik7_c_BWT+RLE", "enwik7_d_BWT+RLE")
        elif choice == 3:
            process_file_nontext_3("idea64.exe", "idea64_c_BWT+RLE", "idea64_d_BWT+RLE.exe")
        elif choice == 4:
            process_file_nontext_3("BlackAndWhite.raw", "BlackAndWhite_c_BWT+RLE.raw", "BlackAndWhite_d_BWT+RLE.raw")
        elif choice == 5:
            process_file_nontext_3("Gray.raw", "Gray_c_BWT+RLE.raw", "Gray_d_BWT+RLE.raw")
        elif choice == 6:
            process_file_nontext_3("Color.raw", "Color_c_BWT+RLE.raw", "Color_d_BWT+RLE.raw")
    elif compressor==4:
        if choice == 1:
            process_file_nontext_4("Кафка Франц.txt", "Кафка Франц_c_BWT+MTF+HA.txt", "Кафка Франц_d_BWT+MTF+HA.txt")
        elif choice == 2:
            process_file_nontext_4("enwik7", "enwik7_c_BWT+MTF+HA", "enwik7_d_BWT+MTF+HA")
        elif choice == 3:
            process_file_nontext_4("idea64.exe", "idea64_c_BWT+MTF+HA", "idea64_d_BWT+MTF+HA.exe")
        elif choice == 4:
            process_file_nontext_4("BlackAndWhite.raw", "BlackAndWhite_c_BWT+MTF+HA.raw", "BlackAndWhite_d_BWT+MTF+HA.raw")
        elif choice == 5:
            process_file_nontext_4("Gray.raw", "Gray_c_BWT+MTF+HA.raw", "Gray_d_BWT+MTF+HA.raw")
        elif choice == 6:
            process_file_nontext_4("Color.raw", "Color_c_BWT+MTF+HA.raw", "Color_d_BWT+MTF+HA.raw")
    elif compressor==5:
        if choice == 1:
            process_file_nontext_5("Кафка Франц.txt", "Кафка Франц_c_BWT+MTF+RLE+HA.txt", "Кафка Франц_d_BWT+MTF+RLE+HA.txt")
        elif choice == 2:
            process_file_nontext_5("enwik7", "enwik7_c_BWT+MTF+RLE+HA", "enwik7_d_BWT+MTF+RLE+HA")
        elif choice == 3:
            process_file_nontext_5("idea64.exe", "idea64_c_BWT+MTF+RLE+HA", "idea64_d_BWT+MTF+RLE+HA.exe")
        elif choice == 4:
            process_file_nontext_5("BlackAndWhite.raw", "BlackAndWhite_c_BWT+MTF+RLE+HA.raw", "BlackAndWhite_d_BWT+MTF+RLE+HA.raw")
        elif choice == 5:
            process_file_nontext_5("Gray.raw", "Gray_c_BWT+MTF+RLE+HA.raw", "Gray_d_BWT+MTF+RLE+HA.raw")
        elif choice == 6:
            process_file_nontext_5("Color.raw", "Color_c_BWT+MTF+RLE+HA.raw", "Color_d_BWT+MTF+RLE+HA.raw")
    elif compressor==6:
        if choice == 1:
            process_file_nontext_6("Кафка Франц.txt", "Кафка Франц_c_ZL77.txt", "Кафка Франц_d_ZL77.txt")
        elif choice == 2:
            process_file_nontext_6("enwik7", "enwik7_c_ZL77", "enwik7_d_ZL77")
        elif choice == 3:
            process_file_nontext_6("idea64.exe", "idea64_c_ZL77", "idea64_d_ZL77.exe")
        elif choice == 4:
            process_file_nontext_6("BlackAndWhite.raw", "BlackAndWhite_c_ZL77.raw", "BlackAndWhite_d_ZL77.raw")
        elif choice == 5:
            process_file_nontext_6("Gray.raw", "Gray_c_ZL77.raw", "Gray_d_ZL77.raw")
        elif choice == 6:
            process_file_nontext_6("Color.raw", "Color_c_ZL77.raw", "Color_d_ZL77.raw")
    elif compressor==7:
        if choice == 1:
            process_file_nontext_7("Кафка Франц.txt", "Кафка Франц_c_ZL77+HA.txt", "Кафка Франц_d_ZL77+HA.txt")
        elif choice == 2:
            process_file_nontext_7("enwik7", "enwik7_c_ZL77+HA", "enwik7_d_ZL77+HA")
        elif choice == 3:
            process_file_nontext_7("idea64.exe", "idea64_c_ZL77+HA", "idea64_d_ZL77+HA.exe")
        elif choice == 4:
            process_file_nontext_7("BlackAndWhite.raw", "BlackAndWhite_c_ZL77+HA.raw", "BlackAndWhite_d_ZL77+HA.raw")
        elif choice == 5:
            process_file_nontext_7("Gray.raw", "Gray_c_ZL77+HA.raw", "Gray_d_ZL77+HA.raw")
        elif choice == 6:
            process_file_nontext_7("Color.raw", "Color_c_ZL77+HA.raw", "Color_d_ZL77+HA.raw")
    elif compressor==8:
        if choice == 1:
            process_file_nontext_8("Кафка Франц.txt", "Кафка Франц_c_LZ78.txt", "Кафка Франц_d_LZ78.txt")
        elif choice == 2:
            process_file_nontext_8("enwik7", "enwik7_c_LZ78", "enwik7_d_LZ78")
        elif choice == 3:
            process_file_nontext_8("idea64.exe", "idea64_c_LZ78", "idea64_d_LZ78.exe")
        elif choice == 4:
            process_file_nontext_8("BlackAndWhite.raw", "BlackAndWhite_c_LZ78.raw", "BlackAndWhite_d_LZ78.raw")
        elif choice == 5:
            process_file_nontext_8("Gray.raw", "Gray_c_LZ78.raw", "Gray_d_LZ78.raw")
        elif choice == 6:
            process_file_nontext_8("Color.raw", "Color_c_LZ78.raw", "Color_d_LZ78.raw")
        elif choice == 7:
            process_file_nontext_8("test.txt", "test_LZ78_c.txt", "test_LZ78_d.txt")
    elif compressor==9:
        if choice == 1:
            process_file_nontext_7("Кафка Франц.txt", "Кафка Франц_c_ZL78+HA.txt", "Кафка Франц_d_ZL78+HA.txt")
        elif choice == 2:
            process_file_nontext_7("enwik7", "enwik7_c_ZL78+HA", "enwik7_d_ZL78+HA")
        elif choice == 3:
            process_file_nontext_7("idea64.exe", "idea64_c_ZL78+HA", "idea64_d_ZL78+HA.exe")
        elif choice == 4:
            process_file_nontext_7("BlackAndWhite.raw", "BlackAndWhite_c_ZL78+HA.raw", "BlackAndWhite_d_ZL78+HA.raw")
        elif choice == 5:
            process_file_nontext_7("Gray.raw", "Gray_c_ZL78+HA.raw", "Gray_d_ZL78+HA.raw")
        elif choice == 6:
            process_file_nontext_7("Color.raw", "Color_c_ZL78+HA.raw", "Color_d_ZL78+HA.raw")