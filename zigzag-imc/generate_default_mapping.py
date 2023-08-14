
def generate_default_mapping(filepath, precision):
    con = """
mapping = {{
    "default": {{
        "core_allocation": 1,
        'operand_precision': {{'O': {pres}, 'O_final': {pres}, 'W': {pres}, 'I': {pres}}},
        'spatial_mapping_hint': {{'D1': ['K', 'OX'], 'D2': ['C','FX','FY'], 'D3':['C', 'FX', 'FY', 'G', 'K','OX','OY']}},
        "memory_operand_links": {{'O': 'O', 'W': 'I1', 'I': 'I2'}},
        'temporal_ordering': [('OX', 'all'), ('OY', 'all'),   ('G', 'all'), ('K', 'all'), ('FX','all'), ('FY','all'), ('C','all')]
    }}
}}
""".format(pres = precision)
    with open(filepath, "w") as f:
        f.write(con)
    # ,'G', 'K','OX','OY'
if __name__ == "__main__":
    generate_default_mapping('../inputs/tinyml/default_mapping.py', 8)
