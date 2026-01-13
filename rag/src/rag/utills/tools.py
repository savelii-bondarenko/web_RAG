def add(a: float, b: float) -> float:
    """Return the sum of two floats.

    Args:
        a (float): first float
        b (float): second float

    Returns:
        float: sum of two floats
    """
    return a + b


def subtract(a: float, b: float) -> float:
    """Return the difference of two floats.

    Args:
        a (float): first float
        b (float): second float

    Returns:
        float: difference of two floats
    """
    return a - b


def multiple(a: float, b: float) -> float:
    """Return the product of two floats.

    Args:
        a (float): first float
        b (float): second float

    Returns:
        float: product of two floats
    """
    return a * b


def divide(a: float, b: float) -> float:
    """Return the quotient of two floats.

    Args:
        a (float): numerator
        b (float): denominator

    Returns:
        float: quotient of a and b

    Raises:
        ValueError: if b is zero
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b


def minimum(nums: list[float]) -> float:
    """Return the minimum element in a list of floats.

    Args:
        nums (list[float]): list of floats

    Returns:
        float: minimum element in the list

    Raises:
        ValueError: if nums is empty
    """
    if not nums:
        raise ValueError("nums must not be empty")
    return min(nums)


def maximum(nums: list[float]) -> float:
    """Return the maximum element in a list of floats.

    Args:
        nums (list[float]): list of floats

    Returns:
        float: maximum element in the list

    Raises:
        ValueError: if nums is empty
    """
    if not nums:
        raise ValueError("nums must not be empty")
    return max(nums)


def average(nums: list[float]) -> float:
    """Return the average of a list of floats.

    Args:
        nums (list[float]): list of floats

    Returns:
        float: average of the list

    Raises:
        ValueError: if nums is empty
    """
    if not nums:
        raise ValueError("nums must not be empty")
    return sum(nums) / len(nums)

TOOLS = [
    add,
    subtract,
    multiple,
    divide,
    minimum,
    maximum,
    average,
]
