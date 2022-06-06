import math


"""
This function calculates how multiple images should be displayed the best on one figure in matplotlib.
It returns a tuple with (num_rows,num_columns)
"""

def get_good_squares(input_data):

    # calculate number of images
    num_images = len(input_data)

    root = int(math.sqrt(num_images))

    # if only one image -> easy
    if num_images == 1:
        return 1,

    # two images are easy
    elif num_images == 2:
        tpl = (1, 2)
        return tpl

    # three images can be put really easy together as well
    elif num_images == 3:
        tpl = (1, 3)
        return tpl

    # if it's a square number, also easy
    elif int(root + 0.5) ** 2 == num_images:

        tpl = (root, root)
        return tpl

    # not a square, there a solution must be found
    else:

        # function to get all divisors:
        def get_divisors(n):
            result = set()
            for i in range(1, int(n ** 0.5) + 1):
                if n % i == 0:
                    result.add(i)
                    result.add(n // i)

            # divisors as list
            divs = list(result)

            # get the counterpart division
            both_divs = []
            for _elem in divs:
                both_divs.append((_elem, int(n/_elem)))

            return both_divs

        # get all divisors
        divisors = get_divisors(num_images)

        # if there's only 2 divisors often it's better to increase the number (11 sucks, 12 is good)
        if len(divisors) == 2:
            divisors = get_divisors(num_images+1)

        # get difference between divisor
        diff_divisor = []
        for elem in divisors:
            diff_divisor.append(abs(elem[0]-elem[1]))

        # find the combination with the smallest difference between the divisors
        min_value = min(diff_divisor)
        min_idx = diff_divisor.index(min_value)
        min_pair = divisors[min_idx]

        return min_pair
