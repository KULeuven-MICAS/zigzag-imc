## Description missing
class Dimension:

    ## The class constructor
    ## @param index: The integer index of this Dimension.
    ## @param name: The user-provided name of this Dimension.
    ## @param size: The user-provided size of this Dimension.
    def __init__(self, index: int, name: str, size: int):
        self.id = index
        self.name = name
        self.size = size

    def __str__(self):
        return f"Dimension(id={self.id},name={self.name},size={self.size})"

    def __repr__(self):
        return str(self)

    ## JSON representation of this class to save it to a json file.
    def __jsonrepr__(self):
        return self.__dict__

    def __eq__(self, other):
        return (
            other.id == self.id and self.name == other.name and self.size == other.size
        )

    def __hash__(self):
        return hash(self.id) ^ hash(self.name)

    def set_array_dim(self, parameter):
        self.size = parameter
