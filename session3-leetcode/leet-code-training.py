import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def title():
    # Inspiration from: https://www.youtube.com/watch?v=RYT08CaYq6A
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Double Pointers (LeetCode Nr. 3)

    Given a string s, find the length of the longest substring without duplicate characters.
    """)
    return


@app.cell
def _():
    string_input = "abcabcbb"
    # output = 3
    return (string_input,)


@app.cell
def _():

    def has_duplicate_characters(s):
        characters = set(s)
        count = {char: s.count(char) for char in characters}
        for k, v in count.items():
            if v > 1:
                return True
        return False

    def lengthOfLongestSubstring(s):
        """
        :type s: str
        :rtype: int
        """
        pointer_left = 0
        pointer_right = 1
        substring = s[pointer_left:pointer_right]
        longest_until_now = 0
        while pointer_right <= len(s):
            if has_duplicate_characters(substring):
                # mantengo el horizonte hacia la derecha, pero muevo el de la izquierda
                pointer_left += 1
            else:
                # extiendo el horizonte hacia la derecha, y mantengo el de la izquierda
                if len(substring) >= longest_until_now:
                    longest_until_now = len(substring)
                    print(f"longest: {longest_until_now}")
                pointer_right += 1
            substring = s[pointer_left:pointer_right]
            print(substring)

        return longest_until_now

    def optimallengthOfLongestSubstring(s):
        """
        The idea: instead of re-checking the whole window for duplicates every time you move a pointer (expensive), keep a dictionary that     remembers where each character was last seen. When you hit a repeat, you can jump the left edge of the window directly past the        earlier occurrence — no scanning required.

        :type s: str
        :rtype: int
        """
        last_seen = {}
        pointer_left = 0
        longest_until_now = 0
        for pointer_right, char in enumerate(s):
            if char in last_seen and last_seen[char] >= pointer_left:
                pointer_left = last_seen[char] + 1
            last_seen[char] = pointer_right
            longest_until_now = max(longest_until_now, pointer_right - pointer_left + 1)

    return (lengthOfLongestSubstring,)


@app.cell
def _(lengthOfLongestSubstring, string_input):
    lengthOfLongestSubstring(string_input)
    return


@app.cell
def _(lengthOfLongestSubstring):
    lengthOfLongestSubstring("pwwkew")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Binary Search (LeetCode Nr.153)

    Suppose an array of length `n` sorted in ascending order is rotated between `1` and `n` times. For example, the array `nums = [0,1,2,4,5,6,7]` might become:

    - `[4,5,6,7,0,1,2]` if it was rotated 4 times.
    - `[0,1,2,4,5,6,7]` if it was rotated 7 times.

    Notice that rotating an array `[a[0], a[1], a[2], ..., a[n-1]]` 1 time results in the array `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]`.

    Given the sorted rotated array nums of unique elements, return the minimum element of this array.

    You must write an algorithm that runs in `O(log n)` time.

    **Example 1:**

    - Input: nums = [3,4,5,1,2]
    - Output: 1

    Explanation: The original array was [1,2,3,4,5] rotated 3 times.

    **Example 2:**

    - Input: nums = [4,5,6,7,0,1,2]
    - Output: 0

    Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.

    **Example 3:**

    - Input: nums = [11,13,15,17]
    - Output: 11

    Explanation: The original array was [11,13,15,17] and it was rotated 4 times.
    """)
    return


@app.cell
def _():
    nums = [11,13,15,17]
    return (nums,)


@app.cell
def _():
    import time

    def is_bigger_than_last(n, list_of_num):
        last = list_of_num[-1]
        return n > last

    def findMin(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        actual_list = nums # list
        min_number = 100000000
        while len(actual_list) > 1:
            print(actual_list)
            time.sleep(2)
            half = len(actual_list) // 2 # index
            actual_value = actual_list[half] # number
            if is_bigger_than_last(actual_value, actual_list): # True
                # the minimum is somewhere in the right
                actual_list = actual_list[half:]
            else: # False
                # the minimum is somewhere on the left (or I'm in the minimum!)
                min_number = min(actual_value, min_number)
                actual_list = actual_list[:half]

        return min(min_number, actual_list[0])



    return findMin, time


@app.cell
def _(findMin, nums):
    findMin(nums)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Breath First Search (LeetCode Nr. 102)

    Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

    - Example 1:
        - Input: root = [3,9,20,null,null,15,7]
        - Output: [[3],[9,20],[15,7]]

    - Example 3:
        - Input: root = [1,2,null,3]
        - Output: [[1],[2],[3]]

    - Example 2:
        - Input: root = [1]
        - Output: [[1]]

    - Example 3:
        - Input: root = []
        - Output: []
    """)
    return


@app.cell
def _():
    values = [3,9,20,None,None,15,7]

    from collections import deque

    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def build_tree(values):
        if not values or values[0] is None:
            return None
        root = TreeNode(values[0])
        queue = deque([root])
        i = 1
        while queue and i < len(values):
            node = queue.popleft()
            if i < len(values):
                left_val = values[i]
                i += 1
                if left_val is not None:
                    node.left = TreeNode(left_val)
                    queue.append(node.left)
            if i < len(values):
                right_val = values[i]
                i += 1
                if right_val is not None:
                    node.right = TreeNode(right_val)
                    queue.append(node.right)
        return root

    root = build_tree(values)
    return deque, root


@app.cell
def _(deque, time):
    def levelOrder(root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[List[int]]
        """
        if not root:
            return []

        level_order = []
        to_visit = deque()
        to_visit.append(root)
        while to_visit:
            # debugging prints:
            print(level_order)
            print(to_visit)
            time.sleep(1)

            # actual code:
            level = []
            n = len(to_visit)
            for _ in range(n):
                node = to_visit.popleft()
                level.append(node.val) # the first value

                if node.left:
                    to_visit.append(node.left)
                if node.right:
                    to_visit.append(node.right)
            level_order.append(level)
        return level_order


    return (levelOrder,)


@app.cell
def _(levelOrder, root):
    levelOrder(root)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Deep First Search (LeetCode Nr. 200, recursion)

    Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

    An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.



    - Example 1:

        - Input: grid = [

          ["1","1","1","1","0"],

          ["1","1","0","1","0"],

          ["1","1","0","0","0"],

          ["0","0","0","0","0"]

        ]
        - Output: 1

    - Example 2:

        - Input: `grid = [

          ["1","1","0","0","0"],

          ["1","1","0","0","0"],

          ["0","0","1","0","0"],

          ["0","0","0","1","1"]

        ]`
        - Output: 3
    """)
    return


@app.cell
def _():
    import copy

    def numIslands(grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """

        n_islands = 0
        grid_copy = copy.deepcopy(grid)
        m = len(grid_copy)
        n = len(grid_copy[0])

        def dfs(i, j):
            # Use recursion to explore: exploring one (i,j) means to explore its neighbors:
            if i >= m or j >= n or i < 0 or j < 0 or grid_copy[i][j] == "0":
                return
            else:
                grid_copy[i][j] = "0"
                dfs(i + 1, j) # going down
                dfs(i, j + 1) # going up
                dfs(i - 1, j) # going 
                dfs(i, j - 1)


        for i in range(m): # rows index
            for j in range(n): # columns index
                if grid_copy[i][j] == "1":
                    n_islands += 1
                    dfs(i, j)

        return n_islands

    return (numIslands,)


@app.cell
def _():
    grid =[
    ["1","1","0","0","1"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["1","0","0","1","1"]
    ]
    return (grid,)


@app.cell
def _(grid, numIslands):
    numIslands(grid)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Back Tracking (LeetCode Nr. 17, combinatoria)

    Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

    A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.


    - Example 1:

        - Input: digits = "23"
        - Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]


    - Example 2:

        - Input: digits = "2"
        - Output: ["a","b","c"]
    """)
    return


@app.cell
def _():
    digits = "23"

    "ab"

    return


@app.cell
def _(mapping):
    def build_substring(substring, digits_to_visit, total_length, mapping):
        if len(substring) == total_length:
            return [substring]  # no necesito seguir
        else:
            d = digits_to_visit[0]  # tomo el primer digito y hago un for loop:
            results = []
            for character in mapping[d]:
                results.extend(build_substring(substring + character, digits_to_visit[1:], total_length, mapping))
            return results


    build_substring("a", "3", 2, mapping)
    return (build_substring,)


@app.cell
def _(build_substring, mapping):



    def letterCombinations(digits, mapping):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []

        mapping = {
        "2": ("a", "b", "c"),
        "3": ("d", "e", "f"),
        "4": ("g", "h", "i"),
        "5": ("j", "k", "l"),
        "6": ("m", "n", "o"),
        "7": ("p", "q", "r", "s"),
        "8": ("t", "u", "v"),
        "9": ("w", "x", "y", "z"),
        }
        return build_substring("", digits, len(digits), mapping)



    letterCombinations("23", mapping)







    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Priority Queue- Top K

    ## LeetCode Nr.215

    Given an integer array `nums` and an integer `nums`, return the kth largest element in the array.

    Note that it is the `k`-th largest element in the sorted order, not the kth distinct element.

    Can you solve it without sorting?


    - Example 1:

        - Input: nums = [3,2,1,5,6,4], k = 2
        - Output: 5

    - Example 2:

        - Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
        - Output: 4
    """)
    return


@app.cell
def _():
    import heapq

    def findKthLargest(nums, k):
        """
        Min Heap (the minimum value is at the top), to find the K largest.

        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # Build a min heap of size k
        min_heap = []
        for num in nums:
            if len(min_heap) < k:
                heapq.heappush(min_heap, num)
            else:
                heapq.heappushpop(min_heap, num)

        return min_heap[0]



    return findKthLargest, heapq


@app.cell
def _(findKthLargest):
    nums = [3,2,3,1,2,4,5,5,6]
    findKthLargest(nums, 4)
    return (nums,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LeetCode Nr. 253

    Given an array of meeting time interval objects consisting of start and end times `[[start_1,end_1],[start_2,end_2],...] (start_i < end_i)`, find the minimum number of rooms required to schedule all meetings without any conflicts.

    Note: (0,8),(8,10) is NOT considered a conflict at 8.

    - Example 1:

        - Input: intervals = [(0,40),(5,10),(15,20)]
        - Output: 2


    Explanation:
    - room1: (0,40)
    - room2: (5,10),(15,20)


    **Solution 1**

    1. Sort all meetings by their start time.
    2. Initialize an empty min heap min_heap to store meeting end times.
    3. Iterate through each meeting in sorted order:
    4. If the heap is not empty and the earliest end time (min_heap[0]) is less than or equal to the current meeting’s start:
        - pop the top of the heap (reuse that room)
    5. Push the current meeting’s end time into the heap (occupy a room).
    6. After processing all meetings:
        - the size of the heap represents the minimum number of rooms required
    7. Return the size of the heap.
    ____________________________________

    **Solution 2**

    1. I sort meetings by start time.
    2. Then I use a min heap to track the end times of active meetings.
    3. For each meeting, I check the earliest ending meeting in the heap:
        - If it ends before the current meeting starts, that room becomes available and I remove it from the heap.
        - Then I add the current meeting's end time.

    The heap size represents the number of rooms currently in use, and the maximum size gives the minimum number of meeting rooms required.
    """)
    return


@app.cell
def _():
    aux = [2,3,4,2]
    aux.sort()
    aux
    return


@app.cell
def _(heapq):

    # Definition of Interval:
    class Interval(object):
        def __init__(self, start, end):
            self.start = start
            self.end = end

    def minMeetingRooms(intervals: list[Interval]) -> int:
        sorted_by_start = sorted(intervals, key=lambda interval: interval.start)

        min_heap = []
        for meeting in intervals:
            if min_heap and min_heap[0] <= meeting.start:
                # The room becames available.
                # Remove the minimum from the heap and push the new meeting.end
                heapq.heappushpop(min_heap, meeting.end)
            else:
                heapq.heappush(min_heap, meeting.end)

        print(min_heap)
        return len(min_heap)



    return Interval, minMeetingRooms


@app.cell
def _(Interval, minMeetingRooms):
    intervals = [Interval(start, end) for start, end in [(0,40),(5,10),(15,20)]]
    minMeetingRooms(intervals)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LeetCode 121: Best time to buy and sell
    https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/

    You are given an array prices where prices[i] is the price of a given stock on the ith day.

    You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

    Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.


    Example 1:

    - Input: prices = [7,1,5,3,6,4]
    - Output: 5
    - Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5. Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.



    Example 2:

    - Input: prices = [7,6,4,3,1]
    - Output: 0
    - Explanation: In this case, no transactions are done and the max profit = 0.
    """)
    return


@app.cell
def _():
    def maximize_profit(array: list):
        """
        Returns the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0
        """
        possible_profit = {i: [] for i in range(len(array))} # day: array of size len(array) - day_index
        for i, value in enumerate(array):
            buy_price = array[i]
            possible_sell_price = {j: array[j] - buy_price for j in range(i, len(array))}
            possible_profit[i] = possible_sell_price

        max_value = 0
        max_index_day = 0
        for k, possible_profit_list in possible_profit.items():
            if max(possible_profit_list) >= max_value:
                max_value = max(possible_profit_list)
                max_index_day = k

        return max_value

    array = [7,1,5,3,6,4]
    maximize_profit(array)
    
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
