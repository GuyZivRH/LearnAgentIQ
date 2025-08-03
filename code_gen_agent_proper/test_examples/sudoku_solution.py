def is_valid_move(puzzle, row, col, num):
    """Check if a number can be placed at a given position."""
    # Check the row
    if num in puzzle[row]:
        return False

    # Check the column
    for i in range(9):
        if puzzle[i][col] == num:
            return False

    # Check the box
    start_row, start_col = row - row % 3, col - col % 3
    for i in range(3):
        for j in range(3):
            if puzzle[i + start_row][j + start_col] == num:
                return False
    return True


def solve_sudoku(puzzle):
    """Solve a Sudoku puzzle using backtracking."""
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] == 0:
                for num in range(1, 10):
                    if is_valid_move(puzzle, i, j, num):
                        puzzle[i][j] = num
                        result = solve_sudoku(puzzle)
                        if result is not None:
                            return result
                        puzzle[i][j] = 0
                return None  # If no number can be placed, backtrack
    return puzzle


def test_solve_sudoku():
    """Test the solve_sudoku function with various cases"""
    
    # Test case 1: Simple solvable puzzle
    puzzle1 = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    
    expected1 = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ]
    
    result1 = solve_sudoku([row[:] for row in puzzle1])  # Create a copy to avoid modifying original puzzle
    assert result1 == expected1, f"Expected solved puzzle, got {result1}"
    print(" Test 1 passed: Simple solvable puzzle")
    
    # Test case 2: Already solved puzzle
    solved_puzzle = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ]
    
    result2 = solve_sudoku([row[:] for row in solved_puzzle])  # Create a copy to avoid modifying original puzzle
    assert result2 == solved_puzzle, f"Expected same puzzle, got {result2}"
    print(" Test 2 passed: Already solved puzzle")
    
    # Test case 3: Unsolvable puzzle (duplicate numbers in row)
    invalid_puzzle = [
        [5, 5, 0, 0, 7, 0, 0, 0, 0],  # Two 5s in first row
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    
    result3 = solve_sudoku([row[:] for row in invalid_puzzle])  # Create a copy to avoid modifying original puzzle
    assert result3 is None, f"Expected None for unsolvable puzzle, got {result3}"
    print(" Test 3 passed: Unsolvable puzzle")
    
    print(" All Sudoku tests passed!")


if __name__ == "__main__":
    test_solve_sudoku()