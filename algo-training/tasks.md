# Tasks

- [Move Zeroes](#move-zeroes)
- [String Compression](#string-compression)
- [Summary Ranges](#summary-ranges)
- [Merge Two Sorted Lists](#merge-two-sorted-lists)


## [Move Zeroes](https://leetcode.com/problems/squares-of-a-sorted-array)

Loop over `nums`. Count non-zero elems and keep this value in `tail`.
If we are ahead of tail (`i > tail`), we can edit the latter.

Time complexity: `O(n)`, where `n` is `len(nums)`  
Memory usage: `O(1)`

```python
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        tail = 0
        for i, el in enumerate(nums):
            if el != 0:
                if i > tail:
                    nums[tail] = el
                tail += 1

        for i in range(1, len(nums) - tail + 1):
            nums[-i] = 0
```


## [String Compression](https://leetcode.com/problems/string-compression/)

Loop over `chars`. Keep the current char and the number it repeats (`cur` and
`curlen`). Not forgetting to convert the number of repetitions to list of
chars.

Time complexity: `O(n)`, where `n` is `len(chars)`  
Memory usage: `O(1)`

```python
class Solution(object):
    def compress(self, chars):
        """
        :type chars: List[str]
        :rtype: int
        """
        cur = chars[0]
        curlen = 0
        pos = 0

        for c in chars:
            if c == cur:
                curlen += 1
            else:
                chars[pos] = cur
                pos += 1
                if curlen > 1:
                    ## Setting slice is slower than setting items 1by1
                    ## https://wiki.python.org/moin/TimeComplexity#list
                    # curlen = str(curlen)
                    # chars[pos:pos+len(curlen)] = list(curlen)
                    # pos += len(curlen)
                    ## O(k+n) vs O(k)
                    for digit in list(str(curlen)):
                        chars[pos] = digit
                        pos += 1
                cur = c
                curlen = 1

        chars[pos] = cur
        pos += 1
        if curlen > 1:
            ## Setting slice is slower than setting items 1by1
            for digit in list(str(curlen)):
                chars[pos] = digit
                pos += 1

        chars = chars[:pos]
        return len(chars)
```


## [Summary Ranges](https://leetcode.com/problems/summary-ranges/)

Loop over `nums`. Check if each next elem grows by 1. If so, that is our
summary range, we are currently on it, otherwise, we get onto another.

Time complexity: `O(n)`, where `n` is `len(nums)`  
Memory usage: `O(n)`

```python
class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        if len(nums) == 0:
            return []

        prev_el = nums[0]
        out = [str(prev_el)]
        trans = False
        el = None

        for el in nums[1:]:
            if el - prev_el == 1:
                trans = True
            else:
                if trans:
                    out[-1] += f"->{prev_el}"
                    trans = False
                out.append(str(el))

            prev_el = el

        ## `el is not None` check is added after pasting to an editor
        if trans and el is not None:
            out[-1] += f"->{el}"

        return out
```


## [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)

Create pre-head nodes (`head` and `path`). To the first of them, we can refer
later - it will show us from where we set off. The second follows the whole
path of the elements sorted in ascending order resulted after merging two
linked lists, without filtration of nodes with the same value.

Time complexity: `O(n)`, where `n` is `len(nums)`  
Memory usage: `O(n)`

```python
# Definition of singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


class Solution:
    def mergeTwoLists(
        self, list1: Optional[ListNode], list2: Optional[ListNode]
    ) -> Optional[ListNode]:

        if list1 is None:
            return list2
        elif list2 is None:
            return list1

        path = ListNode()
        head = ListNode(next=path)

        while list1 is not None and list2 is not None:
            if list1.val == list2.val:
                path.next = list2
                list2 = list2.next
                path = path.next
            elif list1.val > list2.val:
                list1, list2 = list2, list1

            path.next = list1
            list1 = list1.next
            path = path.next

        path.next = list2 if list1 is None else list1
        return head.next.next
```
