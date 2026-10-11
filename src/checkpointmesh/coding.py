import numpy as np

EXP = np.zeros(510, dtype=np.uint8)
LOG = np.zeros(256, dtype=np.int16)
value = 1
for index in range(255):
    EXP[index] = value
    LOG[value] = index
    value <<= 1
    if value & 256:
        value ^= 0x11D
EXP[255:] = EXP[:255]


def multiply(values, coefficient):
    if coefficient == 0:
        return np.zeros_like(values)
    values = np.asarray(values, dtype=np.uint8)
    result = EXP[LOG[values] + LOG[coefficient]].copy()
    result[values == 0] = 0
    return result


def generator(k, r):
    if not 1 <= k <= 255 or r not in [0, 1, 2]:
        raise ValueError("Coding requires 1 <= workers <= 255 and 0 <= parity <= 2")
    matrix = np.eye(k, dtype=np.uint8)
    if r:
        matrix = np.vstack([matrix, np.ones((1, k), dtype=np.uint8)])
    if r == 2:
        matrix = np.vstack([matrix, EXP[:k][None]])
    return matrix


def encode(data, r):
    k = len(data)
    width = max(map(len, data))
    values = [
        np.frombuffer(value.ljust(width, b"\0"), dtype=np.uint8) for value in data
    ]
    result = []
    for row in generator(k, r)[k:]:
        symbol = np.zeros(width, dtype=np.uint8)
        for coefficient, value in zip(row, values):
            symbol ^= multiply(value, int(coefficient))
        result.append(symbol.tobytes())
    return result


def decode(symbols, k, r, lengths):
    if len(symbols) < k:
        raise RuntimeError("Insufficient surviving symbols for stripe recovery")
    indices = sorted(symbols)[:k]
    width = max(len(symbols[i]) for i in indices)
    data = np.stack(
        [
            np.frombuffer(symbols[i].ljust(width, b"\0"), dtype=np.uint8).copy()
            for i in indices
        ]
    )
    matrix = generator(k, r)[indices].copy()
    for column in range(k):
        pivot = next((i for i in range(column, k) if matrix[i, column]), None)
        if pivot is None:
            raise RuntimeError("Coding matrix is singular")
        matrix[[column, pivot]] = matrix[[pivot, column]]
        data[[column, pivot]] = data[[pivot, column]]
        inverse = int(EXP[255 - LOG[int(matrix[column, column])]])
        matrix[column] = multiply(matrix[column], inverse)
        data[column] = multiply(data[column], inverse)
        for row in range(k):
            if row != column and matrix[row, column]:
                coefficient = int(matrix[row, column])
                matrix[row] ^= multiply(matrix[column], coefficient)
                data[row] ^= multiply(data[column], coefficient)
    return [data[i].tobytes()[: lengths[i]] for i in range(k)]
