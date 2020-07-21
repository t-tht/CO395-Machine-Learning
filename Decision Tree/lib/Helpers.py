from collections import defaultdict

# Helper functions used in various modules

# Find the index at which to split a sorted dataset given a threshold
def find_split_point(data, attribute, threshold):
    point = 0

    for d in data:
        if d[attribute] < threshold:
            point += 1
        else:
            break

    return point


# Find the modal value of the decision column
def find_mode(data):
    freq = defaultdict(int)
    max_item = 0
    max_freq = 0
    
    for d in data:
        freq[d[-1]] += 1
        if freq[d[-1]] > max_freq:
            max_item = d[-1]
            max_freq = freq[d[-1]]
    return max_item