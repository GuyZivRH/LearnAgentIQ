# Test cases for largest_rectangle function

def test_largest_rectangle():
    """Test the largest_rectangle function"""
    
    # Test case 1: Empty array should return -1
    assert largest_rectangle([]) == -1, f"Expected -1 for empty array, got {largest_rectangle([])}"
    print("✅ Test 1 passed: Empty array")
    
    # Test case 2: Single element
    assert largest_rectangle([5]) == 5, f"Expected 5 for [5], got {largest_rectangle([5])}"
    print("✅ Test 2 passed: Single element")
    
    # Test case 3: Classic example [2,1,5,6,2,3]
    result = largest_rectangle([2,1,5,6,2,3])
    expected = 10  # Rectangle of height 5 and width 2 (indices 2-3)
    assert result == expected, f"Expected {expected} for [2,1,5,6,2,3], got {result}"
    print("✅ Test 3 passed: Classic example")
    
    # Test case 4: Ascending heights
    assert largest_rectangle([1,2,3,4,5]) == 9, f"Expected 9 for [1,2,3,4,5], got {largest_rectangle([1,2,3,4,5])}"
    print("✅ Test 4 passed: Ascending heights")
    
    # Test case 5: Descending heights  
    assert largest_rectangle([5,4,3,2,1]) == 9, f"Expected 9 for [5,4,3,2,1], got {largest_rectangle([5,4,3,2,1])}"
    print("✅ Test 5 passed: Descending heights")
    
    print("🎉 All tests passed!")

# Run the tests
if __name__ == "__main__":
    test_largest_rectangle() 