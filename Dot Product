# Dot Product

class DotProduct:
    def __init__(self):
        pass
    def dotProduct(self, x, y):
        """
        Calculate the dot product of two vectors x and y.
        :param x: First vector
        :param y: Second vector
        :return: Dot product of x and y
        """
        if len(x) != len(y):
            raise ValueError("Vectors must be of the same length")
        return sum(x_i * y_i for x_i, y_i in zip(x, y, strict=True))
  
if __name__ == "__main__":
        # Example usage
        dp = DotProduct()
        # Example vectors 
        x = [3,2,6]
        y = [1,7,-2]
        result = dp.dotProduct(x, y)
        print(f"The dot product of {x} and {y} is: {result}")
        
