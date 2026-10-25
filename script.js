/**
 * ============================================================================
 * AI Search Visualizer - Core Application Logic
 * ============================================================================
 * An educational pathfinding simulation demonstrating:
 * 1. Breadth-First Search (BFS) using a FIFO Queue
 * 2. Depth-First Search (DFS) using a LIFO Stack
 * 3. Greedy Best-First Search using a Priority Queue with f(n) = h(n)
 * 4. A* Search using a Priority Queue with f(n) = g(n) + h(n)
 *
 * Uses Manhattan distance h(n) = |r1 - r2| + |c1 - c2| for heuristics.
 * ============================================================================
 */

// --- 1. CONFIGURATION & STATE CONSTANTS ---
const GRID_ROWS = 20;
const GRID_COLS = 20;

// Movement directions (Orthogonal 4-directional: Up, Right, Down, Left)
const DIRECTIONS = [
  { r: -1, c: 0, name: 'Up' },
  { r: 0, c: 1, name: 'Right' },
  { r: 1, c: 0, name: 'Down' },
  { r: 0, c: -1, name: 'Left' }
];

// App Global State
const state = {
  grid: [],                 // 2D Array [row][col] storing cell objects
  startPos: { r: 4, c: 4 }, // Default Start coordinate
  goalPos: { r: 15, c: 15 },// Default Goal coordinate
  currentTool: 'wall',      // 'wall', 'erase', 'start', 'goal'
  isMouseDown: false,       // Mouse drag tracking
  drawAction: 'add',        // 'add' or 'remove' for walls during click-drag
  
  // Execution Control
  activeAlgorithm: 'astar', // 'bfs', 'dfs', 'greedy', 'astar'
  searchGenerator: null,    // Active ES6 generator for step-by-step search
  isRunning: false,         // Currently playing animation
  isPaused: false,          // Animation paused
  isFinished: false,        // Search completed
  animationTimer: null,     // Active requestAnimationFrame / timeout handle
  stepDelay: 25,            // Milliseconds per animation step
  
  // Real-time Telemetry Data
  startTime: 0,
  elapsedTime: 0,
  visitedCount: 0,
  frontierCount: 0,
  pathLength: 0,
  pathCost: 0,
  openSetSnapshot: [],      // Nodes currently in OPEN set
  closedSetSnapshot: []     // Nodes currently in CLOSED set
};

// --- 2. PRIORITY QUEUE IMPLEMENTATION ---
/**
 * Min-Priority Queue backed by a binary heap for optimal O(log N) inserts and extractions.
 * Crucial for efficient Greedy Best-First and A* search.
 */
class MinPriorityQueue {
  constructor(comparator = (a, b) => a.f - b.f) {
    this.heap = [];
    this.comparator = comparator;
  }

  size() {
    return this.heap.length;
  }

  isEmpty() {
    return this.heap.length === 0;
  }

  peek() {
    return this.heap[0] || null;
  }

  push(item) {
    this.heap.push(item);
    this._bubbleUp(this.heap.length - 1);
  }

  pop() {
    if (this.isEmpty()) return null;
    const top = this.heap[0];
    const bottom = this.heap.pop();
    if (this.heap.length > 0) {
      this.heap[0] = bottom;
      this._sinkDown(0);
    }
    return top;
  }

  toArray() {
    return [...this.heap];
  }

  _bubbleUp(index) {
    const item = this.heap[index];
    while (index > 0) {
      const parentIdx = Math.floor((index - 1) / 2);
      const parent = this.heap[parentIdx];
      if (this.comparator(item, parent) >= 0) break;
      this.heap[index] = parent;
      index = parentIdx;
    }
    this.heap[index] = item;
  }

  _sinkDown(index) {
    const length = this.heap.length;
    const item = this.heap[index];

    while (true) {
      const leftIdx = 2 * index + 1;
      const rightIdx = 2 * index + 2;
      let swapIdx = null;

      if (leftIdx < length) {
        if (this.comparator(this.heap[leftIdx], item) < 0) {
          swapIdx = leftIdx;
        }
      }

      if (rightIdx < length) {
        if (
          (swapIdx === null && this.comparator(this.heap[rightIdx], item) < 0) ||
          (swapIdx !== null && this.comparator(this.heap[rightIdx], this.heap[leftIdx]) < 0)
        ) {
          swapIdx = rightIdx;
        }
      }

      if (swapIdx === null) break;
      this.heap[index] = this.heap[swapIdx];
      index = swapIdx;
    }
    this.heap[index] = item;
  }
}

// --- 3. HEURISTIC FUNCTIONS ---
/**
 * Computes Manhattan Distance heuristic between two points:
 * h(n) = |r1 - r2| + |c1 - c2|
 */
function manhattanDistance(a, b) {
  return Math.abs(a.r - b.r) + Math.abs(a.c - b.c);
}

// --- 4. GRID INITIALIZATION & DOM RENDERING ---
const gridContainer = document.getElementById('grid-container');

/**
 * Initializes the grid memory structure and renders 400 interactive DOM cells.
 */
function initializeGrid() {
  gridContainer.innerHTML = '';
  state.grid = [];

  for (let r = 0; r < GRID_ROWS; r++) {
    const row = [];
    for (let c = 0; c < GRID_COLS; c++) {
      const cellData = {
        r,
        c,
        isWall: false,
        element: null
      };

      // Create cell element
      const cellEl = document.createElement('div');
      cellEl.className = 'cell';
      cellEl.dataset.r = r;
      cellEl.dataset.c = c;
      cellEl.setAttribute('role', 'gridcell');
      cellEl.setAttribute('aria-label', `Cell row ${r}, column ${c}`);

      // Attach event listeners
      attachCellEvents(cellEl, r, c);

      cellData.element = cellEl;
      row.push(cellData);
      gridContainer.appendChild(cellEl);
    }
    state.grid.push(row);
  }

  // Set default Start and Goal
  renderCell(state.startPos.r, state.startPos.c);
  renderCell(state.goalPos.r, state.goalPos.c);
}

/**
 * Updates DOM visual classes for a specific cell coordinate.
 */
function renderCell(r, c) {
  const cellData = state.grid[r][c];
  const el = cellData.element;
  if (!el) return;

  // Clear transient search classes (keep base cell)
  el.className = 'cell';

  if (r === state.startPos.r && c === state.startPos.c) {
    el.classList.add('cell-start');
  } else if (r === state.goalPos.r && c === state.goalPos.c) {
    el.classList.add('cell-goal');
  } else if (cellData.isWall) {
    el.classList.add('cell-wall');
  }
}

/**
 * Re-renders all cells on the board based on current wall/start/goal state.
 */
function refreshEntireBoard() {
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      renderCell(r, c);
    }
  }
}

// --- 5. INTERACTION & DRAWING HANDLERS ---
function attachCellEvents(el, r, c) {
  // Mouse down: begin drawing or moving
  el.addEventListener('mousedown', (e) => {
    e.preventDefault();
    if (state.isRunning) return; // Prevent board modification during search

    state.isMouseDown = true;
    handleCellInteraction(r, c);
  });

  // Mouse enter: continue drawing walls or erasing when dragging
  el.addEventListener('mouseenter', () => {
    if (!state.isMouseDown || state.isRunning) return;
    handleCellDrag(r, c);
  });
}

// Global mouseup to release drawing mode
window.addEventListener('mouseup', () => {
  state.isMouseDown = false;
});

// Mobile Touch Support: Allows drawing and selecting on touchscreens
gridContainer.addEventListener('touchstart', (e) => {
  if (state.isRunning) return;
  const touch = e.touches[0];
  const target = document.elementFromPoint(touch.clientX, touch.clientY);
  if (target && target.classList.contains('cell')) {
    e.preventDefault();
    state.isMouseDown = true;
    const r = parseInt(target.dataset.r, 10);
    const c = parseInt(target.dataset.c, 10);
    handleCellInteraction(r, c);
  }
}, { passive: false });

gridContainer.addEventListener('touchmove', (e) => {
  if (!state.isMouseDown || state.isRunning) return;
  const touch = e.touches[0];
  const target = document.elementFromPoint(touch.clientX, touch.clientY);
  if (target && target.classList.contains('cell')) {
    e.preventDefault();
    const r = parseInt(target.dataset.r, 10);
    const c = parseInt(target.dataset.c, 10);
    handleCellDrag(r, c);
  }
}, { passive: false });

gridContainer.addEventListener('touchend', () => {
  state.isMouseDown = false;
});

/**
 * Primary click/tap handler for a cell.
 */
function handleCellInteraction(r, c) {
  // If search already ran, reset path before editing grid
  if (state.isFinished || state.searchGenerator) {
    resetPathOnly();
  }

  const isStart = (r === state.startPos.r && c === state.startPos.c);
  const isGoal = (r === state.goalPos.r && c === state.goalPos.c);

  // If clicking on Start or Goal directly, switch active tool to move them
  if (isStart) {
    setDrawingTool('start');
    return;
  }
  if (isGoal) {
    setDrawingTool('goal');
    return;
  }

  // Handle according to active tool
  if (state.currentTool === 'start') {
    if (!isGoal && !state.grid[r][c].isWall) {
      const oldStart = { ...state.startPos };
      state.startPos = { r, c };
      renderCell(oldStart.r, oldStart.c);
      renderCell(r, c);
    }
  } else if (state.currentTool === 'goal') {
    if (!isStart && !state.grid[r][c].isWall) {
      const oldGoal = { ...state.goalPos };
      state.goalPos = { r, c };
      renderCell(oldGoal.r, oldGoal.c);
      renderCell(r, c);
    }
  } else if (state.currentTool === 'erase') {
    state.drawAction = 'remove';
    state.grid[r][c].isWall = false;
    renderCell(r, c);
  } else {
    // 'wall' mode: toggle wall state
    const cell = state.grid[r][c];
    state.drawAction = cell.isWall ? 'remove' : 'add';
    cell.isWall = (state.drawAction === 'add');
    renderCell(r, c);
  }
}

/**
 * Dragging handler for fluid wall painting and erasing.
 */
function handleCellDrag(r, c) {
  const isStart = (r === state.startPos.r && c === state.startPos.c);
  const isGoal = (r === state.goalPos.r && c === state.goalPos.c);

  if (isStart || isGoal) return;

  if (state.currentTool === 'wall') {
    state.grid[r][c].isWall = (state.drawAction === 'add');
    renderCell(r, c);
  } else if (state.currentTool === 'erase') {
    state.grid[r][c].isWall = false;
    renderCell(r, c);
  } else if (state.currentTool === 'start') {
    if (!state.grid[r][c].isWall) {
      const oldStart = { ...state.startPos };
      state.startPos = { r, c };
      renderCell(oldStart.r, oldStart.c);
      renderCell(r, c);
    }
  } else if (state.currentTool === 'goal') {
    if (!state.grid[r][c].isWall) {
      const oldGoal = { ...state.goalPos };
      state.goalPos = { r, c };
      renderCell(oldGoal.r, oldGoal.c);
      renderCell(r, c);
    }
  }
}

/**
 * Changes active drawing tool mode.
 */
function setDrawingTool(tool) {
  state.currentTool = tool;
  document.querySelectorAll('.tool-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tool === tool);
  });
}

// --- 6. MAZE & OBSTACLE GENERATORS ---
/**
 * Clears all walls on the grid.
 */
function clearAllWalls() {
  stopSearchAnimation();
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      state.grid[r][c].isWall = false;
    }
  }
  resetPathOnly();
  refreshEntireBoard();
  updateStatus('Ready. All walls cleared.', 'ready');
}

/**
 * Resets the entire board back to fresh defaults.
 */
function clearEntireBoard() {
  stopSearchAnimation();
  state.startPos = { r: 4, c: 4 };
  state.goalPos = { r: 15, c: 15 };
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      state.grid[r][c].isWall = false;
    }
  }
  resetPathOnly();
  refreshEntireBoard();
  updateStatus('Board cleared to default state.', 'ready');
}

/**
 * Generates random wall obstacles with 30% density.
 */
function generateRandomObstacles() {
  clearAllWalls();
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const isStart = (r === state.startPos.r && c === state.startPos.c);
      const isGoal = (r === state.goalPos.r && c === state.goalPos.c);
      if (!isStart && !isGoal && Math.random() < 0.28) {
        state.grid[r][c].isWall = true;
      }
    }
  }
  refreshEntireBoard();
  updateStatus('Generated random obstacles (28% density).', 'ready');
}

/**
 * Recursive Division Maze Generator.
 * Divides the board recursively with wall chambers and passages.
 */
function generateRecursiveMaze() {
  clearAllWalls();

  // Helper recursive division function
  function divide(rStart, rEnd, cStart, cEnd) {
    const width = cEnd - cStart;
    const height = rEnd - rStart;

    if (width < 3 || height < 3) return;

    // Decide orientation: horizontal or vertical cut
    const horizontal = height > width;

    if (horizontal) {
      // Choose horizontal wall index (even coordinate)
      const possibleRows = [];
      for (let r = rStart + 1; r < rEnd; r++) {
        if (r % 2 === 0) possibleRows.push(r);
      }
      if (possibleRows.length === 0) return;
      const wallRow = possibleRows[Math.floor(Math.random() * possibleRows.length)];

      // Choose a passage gap
      const passageCol = cStart + Math.floor(Math.random() * (cEnd - cStart + 1));

      for (let c = cStart; c <= cEnd; c++) {
        const isStart = (wallRow === state.startPos.r && c === state.startPos.c);
        const isGoal = (wallRow === state.goalPos.r && c === state.goalPos.c);
        if (c !== passageCol && !isStart && !isGoal) {
          state.grid[wallRow][c].isWall = true;
        }
      }

      divide(rStart, wallRow - 1, cStart, cEnd);
      divide(wallRow + 1, rEnd, cStart, cEnd);
    } else {
      // Vertical cut
      const possibleCols = [];
      for (let c = cStart + 1; c < cEnd; c++) {
        if (c % 2 === 0) possibleCols.push(c);
      }
      if (possibleCols.length === 0) return;
      const wallCol = possibleCols[Math.floor(Math.random() * possibleCols.length)];

      // Choose passage gap
      const passageRow = rStart + Math.floor(Math.random() * (rEnd - rStart + 1));

      for (let r = rStart; r <= rEnd; r++) {
        const isStart = (r === state.startPos.r && wallCol === state.startPos.c);
        const isGoal = (r === state.goalPos.r && wallCol === state.goalPos.c);
        if (r !== passageRow && !isStart && !isGoal) {
          state.grid[r][wallCol].isWall = true;
        }
      }

      divide(rStart, rEnd, cStart, wallCol - 1);
      divide(rStart, rEnd, wallCol + 1, cEnd);
    }
  }

  divide(0, GRID_ROWS - 1, 0, GRID_COLS - 1);
  refreshEntireBoard();
  updateStatus('Generated Recursive Division maze with guaranteed passages.', 'ready');
}

/**
 * Generates an interesting spiral obstacle pattern.
 */
function generateSpiralMaze() {
  clearAllWalls();
  let top = 1, bottom = GRID_ROWS - 2, left = 1, right = GRID_COLS - 2;

  while (top <= bottom && left <= right) {
    for (let c = left; c <= right; c++) setWallIfSafe(top, c);
    top += 2;
    for (let r = top - 1; r <= bottom; r++) setWallIfSafe(r, right);
    right -= 2;
    if (top <= bottom) {
      for (let c = right + 1; c >= left; c--) setWallIfSafe(bottom, c);
      bottom -= 2;
    }
    if (left <= right) {
      for (let r = bottom + 1; r >= top; r--) setWallIfSafe(r, left);
      left += 2;
    }
  }

  function setWallIfSafe(r, c) {
    if (r >= 0 && r < GRID_ROWS && c >= 0 && c < GRID_COLS) {
      if ((r !== state.startPos.r || c !== state.startPos.c) &&
          (r !== state.goalPos.r || c !== state.goalPos.c) &&
          (r % 4 !== 0 || c % 3 !== 0)) { // create openings
        state.grid[r][c].isWall = true;
      }
    }
  }

  refreshEntireBoard();
  updateStatus('Generated Spiral maze pattern.', 'ready');
}

// --- 7. SEARCH ALGORITHM GENERATORS ---
/**
 * All 4 search algorithms are formulated as ES6 Generators.
 * This unified structure enables:
 * - Real-time step-by-step animation
 * - Speed control without recreating loops
 * - Single-step stepping execution
 * - Live snapshotting of OPEN and CLOSED sets
 * - Synchronous re-execution for the comparison benchmark!
 */

/**
 * 1. Breadth-First Search (BFS) Generator
 * Data Structure: FIFO Queue (Array shift/push)
 * Guarantee: Optimal shortest path on unweighted grid.
 */
function* bfsGenerator(start, goal, grid) {
  const queue = [{ r: start.r, c: start.c, parent: null, depth: 0 }];
  const visited = new Set();
  const startKey = `${start.r},${start.c}`;
  visited.add(startKey);

  const openSetMap = new Map();
  openSetMap.set(startKey, queue[0]);

  let goalNode = null;

  while (queue.length > 0) {
    const current = queue.shift();
    const currKey = `${current.r},${current.c}`;
    openSetMap.delete(currKey);

    // Yield current expanded node (CLOSED)
    yield {
      type: 'visit',
      current,
      openSet: Array.from(openSetMap.values()),
      closedCount: visited.size
    };

    // Check if goal reached
    if (current.r === goal.r && current.c === goal.c) {
      goalNode = current;
      break;
    }

    // Expand neighbors in standard order
    for (const dir of DIRECTIONS) {
      const nr = current.r + dir.r;
      const nc = current.c + dir.c;
      const nKey = `${nr},${nc}`;

      if (
        nr >= 0 && nr < GRID_ROWS &&
        nc >= 0 && nc < GRID_COLS &&
        !grid[nr][nc].isWall &&
        !visited.has(nKey)
      ) {
        visited.add(nKey);
        const neighborNode = {
          r: nr,
          c: nc,
          parent: current,
          depth: current.depth + 1
        };
        queue.push(neighborNode);
        openSetMap.set(nKey, neighborNode);

        yield {
          type: 'frontier',
          node: neighborNode,
          openSet: Array.from(openSetMap.values())
        };
      }
    }
  }

  // Backtrack path
  yield* reconstructPathGenerator(goalNode, start);
}

/**
 * 2. Depth-First Search (DFS) Generator
 * Data Structure: LIFO Stack (Array push/pop)
 * Guarantee: Not optimal. Explores deepest branch first.
 */
function* dfsGenerator(start, goal, grid) {
  const stack = [{ r: start.r, c: start.c, parent: null, depth: 0 }];
  const visited = new Set();
  const openSetMap = new Map();
  const startKey = `${start.r},${start.c}`;
  openSetMap.set(startKey, stack[0]);

  let goalNode = null;

  while (stack.length > 0) {
    const current = stack.pop();
    const currKey = `${current.r},${current.c}`;
    openSetMap.delete(currKey);

    if (visited.has(currKey)) continue;
    visited.add(currKey);

    yield {
      type: 'visit',
      current,
      openSet: Array.from(openSetMap.values()),
      closedCount: visited.size
    };

    if (current.r === goal.r && current.c === goal.c) {
      goalNode = current;
      break;
    }

    // Push neighbors onto stack (reverse order so Up is explored first when popped)
    const reversedDirs = [...DIRECTIONS].reverse();
    for (const dir of reversedDirs) {
      const nr = current.r + dir.r;
      const nc = current.c + dir.c;
      const nKey = `${nr},${nc}`;

      if (
        nr >= 0 && nr < GRID_ROWS &&
        nc >= 0 && nc < GRID_COLS &&
        !grid[nr][nc].isWall &&
        !visited.has(nKey)
      ) {
        const neighborNode = {
          r: nr,
          c: nc,
          parent: current,
          depth: current.depth + 1
        };
        stack.push(neighborNode);
        openSetMap.set(nKey, neighborNode);

        yield {
          type: 'frontier',
          node: neighborNode,
          openSet: Array.from(openSetMap.values())
        };
      }
    }
  }

  yield* reconstructPathGenerator(goalNode, start);
}

/**
 * 3. Greedy Best-First Search Generator
 * Evaluation Function: f(n) = h(n) (Manhattan distance to goal)
 * Data Structure: Min-Priority Queue
 * Guarantee: Fast heuristic exploration; not guaranteed to find shortest path.
 */
function* greedyGenerator(start, goal, grid) {
  // Sort priority queue by h(n) ascending
  const pq = new MinPriorityQueue((a, b) => a.h - b.h);
  const startH = manhattanDistance(start, goal);
  const startNode = {
    r: start.r,
    c: start.c,
    parent: null,
    g: 0,
    h: startH,
    f: startH
  };

  pq.push(startNode);
  const visited = new Set();
  const inOpenSet = new Map();
  inOpenSet.set(`${start.r},${start.c}`, startNode);

  let goalNode = null;

  while (!pq.isEmpty()) {
    const current = pq.pop();
    const currKey = `${current.r},${current.c}`;
    inOpenSet.delete(currKey);

    if (visited.has(currKey)) continue;
    visited.add(currKey);

    yield {
      type: 'visit',
      current,
      openSet: pq.toArray(),
      closedCount: visited.size
    };

    if (current.r === goal.r && current.c === goal.c) {
      goalNode = current;
      break;
    }

    for (const dir of DIRECTIONS) {
      const nr = current.r + dir.r;
      const nc = current.c + dir.c;
      const nKey = `${nr},${nc}`;

      if (
        nr >= 0 && nr < GRID_ROWS &&
        nc >= 0 && nc < GRID_COLS &&
        !grid[nr][nc].isWall &&
        !visited.has(nKey)
      ) {
        const h = manhattanDistance({ r: nr, c: nc }, goal);
        const neighbor = {
          r: nr,
          c: nc,
          parent: current,
          g: current.g + 1,
          h,
          f: h // In Greedy Best-First, f(n) = h(n)
        };

        pq.push(neighbor);
        inOpenSet.set(nKey, neighbor);

        yield {
          type: 'frontier',
          node: neighbor,
          openSet: pq.toArray()
        };
      }
    }
  }

  yield* reconstructPathGenerator(goalNode, start);
}

/**
 * 4. A* Search Generator
 * Evaluation Function: f(n) = g(n) + h(n)
 * Data Structure: Min-Priority Queue
 * Guarantee: Optimal shortest path on unweighted grid.
 */
function* astarGenerator(start, goal, grid) {
  // Sort priority queue by f(n) ascending; tie-break with lower h(n)
  const pq = new MinPriorityQueue((a, b) => {
    if (a.f !== b.f) return a.f - b.f;
    return a.h - b.h; // Tie-breaker: prefer closer to goal
  });

  const startH = manhattanDistance(start, goal);
  const startNode = {
    r: start.r,
    c: start.c,
    parent: null,
    g: 0,
    h: startH,
    f: startH
  };

  pq.push(startNode);

  // gScores map: lowest known cost from start to node (r,c)
  const gScore = new Map();
  const startKey = `${start.r},${start.c}`;
  gScore.set(startKey, 0);

  const visited = new Set();
  let goalNode = null;

  while (!pq.isEmpty()) {
    const current = pq.pop();
    const currKey = `${current.r},${current.c}`;

    // Skip if we already expanded this coordinate with a better/equal g-score
    if (visited.has(currKey)) continue;
    visited.add(currKey);

    yield {
      type: 'visit',
      current,
      openSet: pq.toArray(),
      closedCount: visited.size
    };

    if (current.r === goal.r && current.c === goal.c) {
      goalNode = current;
      break;
    }

    for (const dir of DIRECTIONS) {
      const nr = current.r + dir.r;
      const nc = current.c + dir.c;
      const nKey = `${nr},${nc}`;

      if (
        nr >= 0 && nr < GRID_ROWS &&
        nc >= 0 && nc < GRID_COLS &&
        !grid[nr][nc].isWall
      ) {
        const tentativeG = current.g + 1;
        const currentBestG = gScore.has(nKey) ? gScore.get(nKey) : Infinity;

        if (tentativeG < currentBestG) {
          gScore.set(nKey, tentativeG);
          const h = manhattanDistance({ r: nr, c: nc }, goal);
          const neighbor = {
            r: nr,
            c: nc,
            parent: current,
            g: tentativeG,
            h,
            f: tentativeG + h // f(n) = g(n) + h(n)
          };

          pq.push(neighbor);

          yield {
            type: 'frontier',
            node: neighbor,
            openSet: pq.toArray()
          };
        }
      }
    }
  }

  yield* reconstructPathGenerator(goalNode, start);
}

/**
 * Helper generator to backtrack and yield the final optimal path nodes.
 */
function* reconstructPathGenerator(goalNode, start) {
  if (!goalNode) {
    yield { type: 'no-path' };
    return;
  }

  const path = [];
  let curr = goalNode;
  while (curr) {
    path.push(curr);
    curr = curr.parent;
  }
  path.reverse(); // Start to Goal

  // Yield path reconstruction step by step
  for (const node of path) {
    yield {
      type: 'path-step',
      node,
      pathLength: path.length - 1, // Number of moves
      pathCost: path.length - 1
    };
  }

  yield {
    type: 'done',
    pathLength: path.length - 1,
    pathCost: path.length - 1
  };
}

// --- 8. ANIMATION LOOP & STEP CONTROLLER ---
/**
 * Stops any ongoing search animation timer.
 */
function stopSearchAnimation() {
  if (state.animationTimer) {
    clearTimeout(state.animationTimer);
    state.animationTimer = null;
  }
  state.isRunning = false;
  state.isPaused = false;
  updatePlayButtonUI();
}

/**
 * Resets search paths, frontiers, and visited states, preserving walls.
 */
function resetPathOnly() {
  stopSearchAnimation();
  state.searchGenerator = null;
  state.isFinished = false;

  // Clear visual classes from all cells
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const el = state.grid[r][c].element;
      if (el) {
        el.classList.remove('cell-visited', 'cell-frontier', 'cell-path');
      }
    }
  }

  // Reset telemetry stats
  state.visitedCount = 0;
  state.frontierCount = 0;
  state.pathLength = 0;
  state.pathCost = 0;
  state.openSetSnapshot = [];
  state.closedSetSnapshot = [];

  document.getElementById('stat-visited').textContent = '0';
  document.getElementById('stat-frontier').textContent = '0';
  document.getElementById('stat-path-len').textContent = '-';
  document.getElementById('stat-cost').textContent = '-';
  document.getElementById('stat-time').textContent = '0 ms';
  document.getElementById('open-count').textContent = '0';
  document.getElementById('closed-count').textContent = '0';

  renderOpenSetInspector([]);
  renderClosedSetInspector([]);

  updateStatus('Ready. Choose algorithm and click Visualize Search.', 'ready');
}

/**
 * Starts or resumes search visualization.
 */
function startOrResumeSearch() {
  if (state.isRunning) {
    // Pause action
    state.isRunning = false;
    state.isPaused = true;
    updatePlayButtonUI();
    updateStatus('Search paused. Click Resume or Step.', 'ready');
    return;
  }

  if (state.isFinished) {
    resetPathOnly();
  }

  // Initialize generator if not already active
  if (!state.searchGenerator) {
    resetPathOnly();
    state.startTime = performance.now();
    state.searchGenerator = getActiveGenerator();
  }

  state.isRunning = true;
  state.isPaused = false;
  updatePlayButtonUI();
  updateStatus(`Searching using ${getAlgorithmDisplayName(state.activeAlgorithm)}...`, 'searching');

  runAnimationLoop();
}

/**
 * Returns generator instance for selected algorithm.
 */
function getActiveGenerator() {
  switch (state.activeAlgorithm) {
    case 'bfs':
      return bfsGenerator(state.startPos, state.goalPos, state.grid);
    case 'dfs':
      return dfsGenerator(state.startPos, state.goalPos, state.grid);
    case 'greedy':
      return greedyGenerator(state.startPos, state.goalPos, state.grid);
    case 'astar':
    default:
      return astarGenerator(state.startPos, state.goalPos, state.grid);
  }
}

/**
 * Asynchronous animation step driver.
 */
function runAnimationLoop() {
  if (!state.isRunning) return;

  const stepResult = executeSingleStep();

  if (stepResult.done) {
    state.isRunning = false;
    state.isPaused = false;
    state.isFinished = true;
    updatePlayButtonUI();
    return;
  }

  // Schedule next step based on slider delay
  state.animationTimer = setTimeout(() => {
    runAnimationLoop();
  }, state.stepDelay);
}

/**
 * Executes a single discrete step from the algorithm generator.
 */
function executeSingleStep() {
  if (!state.searchGenerator) {
    state.startTime = performance.now();
    state.searchGenerator = getActiveGenerator();
  }

  const next = state.searchGenerator.next();
  if (next.done) {
    return { done: true };
  }

  const val = next.value;
  applyStepResult(val);
  return { done: false, value: val };
}

/**
 * Applies step result to DOM elements and live telemetry.
 */
function applyStepResult(step) {
  const isStart = (node) => node.r === state.startPos.r && node.c === state.startPos.c;
  const isGoal = (node) => node.r === state.goalPos.r && node.c === state.goalPos.c;

  // Track elapsed time
  state.elapsedTime = Math.round(performance.now() - state.startTime);
  document.getElementById('stat-time').textContent = `${state.elapsedTime} ms`;

  switch (step.type) {
    case 'frontier': {
      const node = step.node;
      if (!isStart(node) && !isGoal(node)) {
        const el = state.grid[node.r][node.c].element;
        if (!el.classList.contains('cell-visited')) {
          el.classList.add('cell-frontier');
        }
      }
      state.frontierCount = step.openSet.length;
      document.getElementById('stat-frontier').textContent = state.frontierCount;
      document.getElementById('open-count').textContent = state.frontierCount;
      renderOpenSetInspector(step.openSet);
      break;
    }

    case 'visit': {
      const node = step.current;
      if (!isStart(node) && !isGoal(node)) {
        const el = state.grid[node.r][node.c].element;
        el.classList.remove('cell-frontier');
        el.classList.add('cell-visited');
      }
      state.visitedCount = step.closedCount;
      document.getElementById('stat-visited').textContent = state.visitedCount;
      document.getElementById('closed-count').textContent = state.visitedCount;

      state.closedSetSnapshot.push(node);
      renderClosedSetInspector(state.closedSetSnapshot);

      if (step.openSet) {
        state.frontierCount = step.openSet.length;
        document.getElementById('stat-frontier').textContent = state.frontierCount;
        document.getElementById('open-count').textContent = state.frontierCount;
        renderOpenSetInspector(step.openSet);
      }
      break;
    }

    case 'path-step': {
      const node = step.node;
      if (!isStart(node) && !isGoal(node)) {
        const el = state.grid[node.r][node.c].element;
        el.classList.remove('cell-visited', 'cell-frontier');
        el.classList.add('cell-path');
      }
      break;
    }

    case 'no-path': {
      updateStatus('Search complete: No path exists to the Goal.', 'fail');
      document.getElementById('stat-path-len').textContent = 'N/A';
      document.getElementById('stat-cost').textContent = 'N/A';
      break;
    }

    case 'done': {
      state.pathLength = step.pathLength;
      state.pathCost = step.pathCost;
      document.getElementById('stat-path-len').textContent = state.pathLength;
      document.getElementById('stat-cost').textContent = state.pathCost;
      updateStatus(
        `Path Found! Length: ${step.pathLength} steps • Cost: ${step.pathCost} • Nodes Visited: ${state.visitedCount}`,
        'success'
      );
      break;
    }
  }
}

/**
 * Updates primary Visualize / Pause button appearance.
 */
function updatePlayButtonUI() {
  const btn = document.getElementById('btn-start');
  const btnText = document.getElementById('btn-start-text');
  const icon = document.getElementById('play-icon');

  if (state.isRunning) {
    btnText.textContent = 'Pause';
    icon.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
    btn.classList.add('btn-secondary');
    btn.classList.remove('btn-primary');
  } else if (state.isPaused) {
    btnText.textContent = 'Resume';
    icon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
    btn.classList.add('btn-primary');
    btn.classList.remove('btn-secondary');
  } else {
    btnText.textContent = 'Visualize Search';
    icon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
    btn.classList.add('btn-primary');
    btn.classList.remove('btn-secondary');
  }
}

/**
 * Updates status bar banner message and color state.
 */
function updateStatus(message, type = 'ready') {
  const bar = document.getElementById('status-bar');
  const text = document.getElementById('status-text');
  bar.className = `status-bar ${type}`;
  text.innerHTML = message;
}

// --- 9. OPEN & CLOSED SETS INSPECTOR RENDERING ---
/**
 * Renders the OPEN set nodes in the inspector panel.
 */
function renderOpenSetInspector(nodes) {
  const container = document.getElementById('open-list-container');
  if (!nodes || nodes.length === 0) {
    container.innerHTML = '<p class="empty-hint">OPEN set is currently empty.</p>';
    return;
  }

  // Display top 24 nodes to prevent DOM congestion
  const displayNodes = nodes.slice(0, 24);
  let html = '';

  for (const n of displayNodes) {
    let metricText = '';
    if (state.activeAlgorithm === 'astar') {
      metricText = `<span class="chip-f">f=${n.f} (g=${n.g}, h=${n.h})</span>`;
    } else if (state.activeAlgorithm === 'greedy') {
      metricText = `<span class="chip-f">h=${n.h}</span>`;
    } else {
      metricText = `<span class="chip-f">d=${n.depth || 0}</span>`;
    }

    html += `
      <div class="node-chip chip-frontier" title="Node (${n.r}, ${n.c})">
        <span class="chip-coord">(${n.r}, ${n.c})</span>
        ${metricText}
      </div>
    `;
  }

  if (nodes.length > 24) {
    html += `<div class="node-chip chip-more">+${nodes.length - 24} more...</div>`;
  }

  container.innerHTML = html;
}

/**
 * Renders the CLOSED set nodes in the inspector panel.
 */
function renderClosedSetInspector(nodes) {
  const container = document.getElementById('closed-list-container');
  if (!nodes || nodes.length === 0) {
    container.innerHTML = '<p class="empty-hint">CLOSED set is currently empty.</p>';
    return;
  }

  // Show the most recent 24 nodes added to CLOSED set
  const recentNodes = nodes.slice(-24).reverse();
  let html = '';

  for (const n of recentNodes) {
    html += `
      <div class="node-chip chip-visited" title="Expanded node (${n.r}, ${n.c})">
        <span class="chip-coord">(${n.r}, ${n.c})</span>
        <span class="chip-f">✓</span>
      </div>
    `;
  }

  if (nodes.length > 24) {
    html += `<div class="node-chip chip-more">Total ${nodes.length} visited</div>`;
  }

  container.innerHTML = html;
}

// --- 10. ALGORITHM BENCHMARK & COMPARISON MODAL ---
/**
 * Runs all 4 algorithms synchronously on identical clone of current grid.
 */
function runComparisonBenchmark() {
  stopSearchAnimation();

  const algorithms = [
    { id: 'bfs', name: 'BFS', ds: 'Queue (FIFO)', formula: 'f(n) = depth(n)', isOptimal: true },
    { id: 'dfs', name: 'DFS', ds: 'Stack (LIFO)', formula: 'f(n) = -depth(n)', isOptimal: false },
    { id: 'greedy', name: 'Greedy Best-First', ds: 'Priority Queue', formula: 'f(n) = h(n)', isOptimal: false },
    { id: 'astar', name: 'A* Search', ds: 'Priority Queue', formula: 'f(n) = g(n) + h(n)', isOptimal: true }
  ];

  const results = [];

  for (const algo of algorithms) {
    const t0 = performance.now();
    let gen = null;

    if (algo.id === 'bfs') gen = bfsGenerator(state.startPos, state.goalPos, state.grid);
    else if (algo.id === 'dfs') gen = dfsGenerator(state.startPos, state.goalPos, state.grid);
    else if (algo.id === 'greedy') gen = greedyGenerator(state.startPos, state.goalPos, state.grid);
    else gen = astarGenerator(state.startPos, state.goalPos, state.grid);

    let visitedCount = 0;
    let pathFound = false;
    let pathLength = 0;
    let pathCost = 0;

    let step = gen.next();
    while (!step.done) {
      const v = step.value;
      if (v.type === 'visit') {
        visitedCount = v.closedCount;
      } else if (v.type === 'done') {
        pathFound = true;
        pathLength = v.pathLength;
        pathCost = v.pathCost;
      }
      step = gen.next();
    }

    const t1 = performance.now();
    const durationMs = Math.round((t1 - t0) * 100) / 100;

    results.push({
      ...algo,
      visitedCount,
      pathFound,
      pathLength: pathFound ? pathLength : 'N/A',
      pathCost: pathFound ? pathCost : 'N/A',
      durationMs
    });
  }

  displayComparisonModal(results);
}

/**
 * Displays modal with results table and relative efficiency bars.
 */
function displayComparisonModal(results) {
  const modal = document.getElementById('compare-modal');
  const tbody = document.getElementById('comparison-table-body');
  const barsContainer = document.getElementById('compare-chart-bars');

  tbody.innerHTML = '';
  barsContainer.innerHTML = '';

  const maxVisited = Math.max(...results.map(r => r.visitedCount), 1);

  results.forEach(res => {
    // Populate Table Row
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>
        <span class="algo-cell-name">
          <span class="legend-badge bar-${res.id}"></span>
          ${res.name}
        </span>
      </td>
      <td><code>${res.ds}</code></td>
      <td><code>${res.formula}</code></td>
      <td class="font-mono"><strong>${res.visitedCount}</strong></td>
      <td class="font-mono">${res.pathLength}</td>
      <td class="font-mono">${res.pathCost}</td>
      <td>
        <span class="badge-status ${res.isOptimal ? 'yes' : 'no'}">
          ${res.isOptimal ? 'Yes (Optimal)' : 'No'}
        </span>
      </td>
      <td class="font-mono">${res.durationMs} ms</td>
    `;
    tbody.appendChild(tr);

    // Populate Comparison Bar
    const percent = Math.max((res.visitedCount / maxVisited) * 100, 3);
    const barRow = document.createElement('div');
    barRow.className = 'chart-bar-row';
    barRow.innerHTML = `
      <div class="bar-meta">
        <span class="bar-name">${res.name} (${res.visitedCount} visited)</span>
        <span class="bar-value">Path Length: ${res.pathLength}</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill bar-${res.id}" style="width: ${percent}%;"></div>
      </div>
    `;
    barsContainer.appendChild(barRow);
  });

  modal.classList.add('open');
  modal.setAttribute('aria-hidden', 'false');
}

function closeComparisonModal() {
  const modal = document.getElementById('compare-modal');
  modal.classList.remove('open');
  modal.setAttribute('aria-hidden', 'true');
}

// --- 11. ALGORITHM THEORY & TREE VISUALIZATION CONTENT ---
const theoryData = {
  bfs: {
    title: 'Breadth-First Search (BFS)',
    category: 'Uninformed Search',
    dataStructure: 'Queue (FIFO: First-In, First-Out)',
    formula: 'f(n) = depth(n)',
    definition: 'BFS traverses the search graph level by level, thoroughly examining all nodes at distance k before moving to nodes at distance k+1. It guarantees finding the shortest path in unweighted graphs with uniform edge costs.',
    optimality: 'Guaranteed optimal for uniform-cost / unweighted step graphs.',
    completeness: 'Complete if the branching factor b is finite.',
    timeComplexity: 'O(b^d) where b is branching factor and d is goal depth.',
    spaceComplexity: 'O(b^d) keeps every node in memory, making memory the primary bottleneck.',
    treeSvg: generateTreeSvg('bfs'),
    treeExplanation: [
      '<strong>Step 1:</strong> Start at Root <strong>A</strong>. OPEN = [A], CLOSED = [].',
      '<strong>Step 2:</strong> Expand <strong>A</strong>. Add children B, C to FIFO Queue. OPEN = [B, C], CLOSED = [A].',
      '<strong>Step 3:</strong> Expand <strong>B</strong>. Add children D, E to Queue back. OPEN = [C, D, E], CLOSED = [A, B].',
      '<strong>Step 4:</strong> Expand <strong>C</strong>. Add children F, G. OPEN = [D, E, F, G], CLOSED = [A, B, C].',
      '<strong>Step 5:</strong> Nodes at depth 2 (D, E, F, G) are evaluated. When Goal <strong>G</strong> is reached, path <code>A → C → G</code> is returned (length 2).'
    ]
  },
  dfs: {
    title: 'Depth-First Search (DFS)',
    category: 'Uninformed Search',
    dataStructure: 'Stack (LIFO: Last-In, First-Out)',
    formula: 'f(n) = -depth(n)',
    definition: 'DFS explores as deep as possible along each branch before backtracking. It plunges down a chosen branch until reaching a dead end or the goal, then backtracks to explore alternate branches.',
    optimality: 'Not optimal. Can find long, winding paths even when a direct 1-step path exists.',
    completeness: 'Complete on finite graphs with cycle detection; incomplete on infinite graphs.',
    timeComplexity: 'O(b^m) where m is maximum search depth.',
    spaceComplexity: 'O(b · m) very space efficient since only the current path is kept in the stack.',
    treeSvg: generateTreeSvg('dfs'),
    treeExplanation: [
      '<strong>Step 1:</strong> Push Root <strong>A</strong> to LIFO Stack. OPEN = [A].',
      '<strong>Step 2:</strong> Pop <strong>A</strong>. Push C, then B. Top of Stack is <strong>B</strong>. OPEN = [C, B].',
      '<strong>Step 3:</strong> Pop <strong>B</strong>. Push E, then D. Top of Stack is <strong>D</strong>. OPEN = [C, E, D].',
      '<strong>Step 4:</strong> Pop <strong>D</strong> (leaf). Backtracks and pops <strong>E</strong>.',
      '<strong>Step 5:</strong> Only after the entire left subtree (B, D, E) is exhausted does DFS backtrack to <strong>C</strong> and reach Goal <strong>G</strong>, resulting in a potentially suboptimal search trajectory.'
    ]
  },
  greedy: {
    title: 'Greedy Best-First Search',
    category: 'Informed (Heuristic) Search',
    dataStructure: 'Min-Priority Queue',
    formula: 'f(n) = h(n)',
    definition: 'Greedy Best-First search evaluates candidate nodes purely by their heuristic estimate to the goal, ignoring accumulated path cost g(n). It always expands the node that appears closest to the goal.',
    optimality: 'Not optimal. Because it ignores past path cost g(n), it can be lured down dead ends or trapped in obstacle pockets.',
    completeness: 'Incomplete in infinite spaces without cycle prevention; complete with graph search.',
    timeComplexity: 'O(b^m) in worst case; often very fast with an accurate heuristic.',
    spaceComplexity: 'O(b^m) maintains all explored nodes in memory.',
    treeSvg: generateTreeSvg('greedy'),
    treeExplanation: [
      '<strong>Sample Heuristics:</strong> h(A)=6, h(B)=4, h(C)=2, h(D)=5, h(E)=4, h(F)=3, h(G)=0 (Goal).',
      '<strong>Step 1:</strong> Start at <strong>A</strong> (f=6). OPEN = [A].',
      '<strong>Step 2:</strong> Expand <strong>A</strong>. Insert B (f=4) and C (f=2). Since h(C) < h(B), <strong>C</strong> is top priority! OPEN = [C(2), B(4)].',
      '<strong>Step 3:</strong> Expand <strong>C</strong> directly, skipping the entire B subtree! Children are F(3) and G(0).',
      '<strong>Step 4:</strong> Expand <strong>G</strong> (f=0). Goal is reached immediately because heuristic guidance directed search directly toward G.'
    ]
  },
  astar: {
    title: 'A* Search (A-Star)',
    category: 'Informed (Heuristic) Search',
    dataStructure: 'Min-Priority Queue',
    formula: 'f(n) = g(n) + h(n)',
    definition: 'A* combines the proven optimality of Dijkstra (uniform-cost search using g(n)) with the goal-directed speed of Greedy Best-First (using h(n)). With an admissible and consistent heuristic (like Manhattan distance on a grid), A* is mathematically guaranteed to find the optimal shortest path while visiting the minimum necessary states.',
    optimality: 'Provably optimal with an admissible (never overestimating) heuristic.',
    completeness: 'Complete: guaranteed to find a path if one exists.',
    timeComplexity: 'O(b^d) in worst case; practically O(b^ε) with good heuristic.',
    spaceComplexity: 'O(b^d) retains all generated nodes in memory.',
    treeSvg: generateTreeSvg('astar'),
    treeExplanation: [
      '<strong>Heuristic & Edge Costs:</strong> Uniform step cost g=1 each step. h(A)=6, h(B)=4, h(C)=2, h(D)=5, h(E)=4, h(F)=3, h(G)=0.',
      '<strong>Step 1:</strong> Start at <strong>A</strong>: g=0, h=6 ⇒ f(A) = 6.',
      '<strong>Step 2:</strong> Expand <strong>A</strong>: B has g=1, h=4 ⇒ f(B)=5. C has g=1, h=2 ⇒ f(C)=3. OPEN = [C(f=3), B(f=5)].',
      '<strong>Step 3:</strong> Node <strong>C</strong> has lowest f(n)=3. Expand C: F has g=2, h=3 ⇒ f(F)=5. G has g=2, h=0 ⇒ f(G)=2. OPEN = [G(f=2), B(f=5), F(f=5)].',
      '<strong>Step 4:</strong> Node <strong>G</strong> has lowest f(n)=2. Goal is expanded! Optimal path <code>A → C → G</code> found with total cost 2.'
    ]
  }
};

/**
 * Dynamically generates a clean, readable SVG diagram of tree A → B, C → D, E, F, G.
 */
function generateTreeSvg(algoId) {
  return `
    <svg class="tree-svg" viewBox="0 0 340 180" width="100%" height="160">
      <!-- Edges -->
      <line x1="170" y1="28" x2="90" y2="80" class="tree-edge" />
      <line x1="170" y1="28" x2="250" y2="80" class="tree-edge" />
      <line x1="90" y1="80" x2="50" y2="140" class="tree-edge" />
      <line x1="90" y1="80" x2="130" y2="140" class="tree-edge" />
      <line x1="250" y1="80" x2="210" y2="140" class="tree-edge" />
      <line x1="250" y1="80" x2="290" y2="140" class="tree-edge" />

      <!-- Node A (Root) -->
      <circle cx="170" cy="28" r="16" class="tree-node-circle root-node" />
      <text x="170" y="28" class="tree-text">A</text>
      <text x="170" y="5" class="tree-cost-label">${algoId === 'astar' ? 'f=6' : algoId === 'greedy' ? 'h=6' : 'd=0'}</text>

      <!-- Level 1: B & C -->
      <circle cx="90" cy="80" r="16" class="tree-node-circle" />
      <text x="90" y="80" class="tree-text">B</text>
      <text x="90" y="60" class="tree-cost-label">${algoId === 'astar' ? 'f=5' : algoId === 'greedy' ? 'h=4' : 'd=1'}</text>

      <circle cx="250" cy="80" r="16" class="tree-node-circle" />
      <text x="250" y="80" class="tree-text">C</text>
      <text x="250" y="60" class="tree-cost-label">${algoId === 'astar' ? 'f=3' : algoId === 'greedy' ? 'h=2' : 'd=1'}</text>

      <!-- Level 2: D, E, F, G -->
      <circle cx="50" cy="140" r="15" class="tree-node-circle" />
      <text x="50" y="140" class="tree-text">D</text>
      <text x="50" y="165" class="tree-cost-label">${algoId === 'astar' ? 'f=7' : algoId === 'greedy' ? 'h=5' : 'd=2'}</text>

      <circle cx="130" cy="140" r="15" class="tree-node-circle" />
      <text x="130" y="140" class="tree-text">E</text>
      <text x="130" y="165" class="tree-cost-label">${algoId === 'astar' ? 'f=6' : algoId === 'greedy' ? 'h=4' : 'd=2'}</text>

      <circle cx="210" cy="140" r="15" class="tree-node-circle" />
      <text x="210" y="140" class="tree-text">F</text>
      <text x="210" y="165" class="tree-cost-label">${algoId === 'astar' ? 'f=5' : algoId === 'greedy' ? 'h=3' : 'd=2'}</text>

      <!-- Node G (Goal) -->
      <circle cx="290" cy="140" r="15" class="tree-node-circle goal-node" />
      <text x="290" y="140" class="tree-text">G*</text>
      <text x="290" y="165" class="tree-cost-label">${algoId === 'astar' ? 'f=2' : algoId === 'greedy' ? 'h=0' : 'd=2'}</text>
    </svg>
  `;
}

/**
 * Renders the Theory section based on selected algorithm tab.
 */
function renderTheorySection(algoId) {
  const container = document.getElementById('theory-content-box');
  const data = theoryData[algoId];
  if (!data) return;

  const walkthroughHtml = data.treeExplanation
    .map(step => `<p class="walkthrough-step">${step}</p>`)
    .join('');

  container.innerHTML = `
    <div class="theory-card-grid">
      <!-- Left: Conceptual Info & Properties -->
      <div class="theory-info-pane">
        <div class="theory-badge-row">
          <span class="pill-badge highlight">${data.category}</span>
          <span class="pill-badge">${data.dataStructure}</span>
        </div>

        <p class="theory-def">${data.definition}</p>

        <div class="formula-block">
          Evaluation Function: <code>${data.formula}</code>
        </div>

        <table class="theory-detail-table">
          <tbody>
            <tr>
              <th>Data Structure</th>
              <td><code>${data.dataStructure}</code></td>
            </tr>
            <tr>
              <th>Optimality</th>
              <td>${data.optimality}</td>
            </tr>
            <tr>
              <th>Completeness</th>
              <td>${data.completeness}</td>
            </tr>
            <tr>
              <th>Time Complexity</th>
              <td><code>${data.timeComplexity}</code></td>
            </tr>
            <tr>
              <th>Space Complexity</th>
              <td><code>${data.spaceComplexity}</code></td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Right: Example Search Tree -->
      <div class="theory-tree-pane">
        <div class="tree-heading">
          <span>Search Tree Example (A → B, C → D, E, F, G)</span>
          <span class="pill-badge">Goal = Node G</span>
        </div>

        <div class="tree-canvas-wrapper">
          ${data.treeSvg}
        </div>

        <div class="tree-walkthrough">
          ${walkthroughHtml}
        </div>
      </div>
    </div>
  `;
}

// --- 12. HELPER UI MAPPINGS ---
function getAlgorithmDisplayName(id) {
  switch (id) {
    case 'bfs': return 'Breadth-First Search (BFS)';
    case 'dfs': return 'Depth-First Search (DFS)';
    case 'greedy': return 'Greedy Best-First Search';
    case 'astar': return 'A* Search (A-Star)';
    default: return id;
  }
}

function updateFormulaPreview() {
  const preview = document.getElementById('formula-preview');
  const optBadge = document.getElementById('stat-optimal');
  const openOrderHint = document.getElementById('open-order-hint');

  switch (state.activeAlgorithm) {
    case 'bfs':
      preview.textContent = 'BFS: f(n) = depth(n)';
      optBadge.textContent = 'Optimal (Unweighted)';
      optBadge.className = 'metric-badge optimal';
      openOrderHint.textContent = 'FIFO: First-In First-Out';
      break;
    case 'dfs':
      preview.textContent = 'DFS: f(n) = -depth(n)';
      optBadge.textContent = 'Suboptimal';
      optBadge.className = 'metric-badge suboptimal';
      openOrderHint.textContent = 'LIFO: Last-In First-Out';
      break;
    case 'greedy':
      preview.textContent = 'Greedy: f(n) = h(n)';
      optBadge.textContent = 'Suboptimal (Heuristic)';
      optBadge.className = 'metric-badge suboptimal';
      openOrderHint.textContent = 'Sorted by h(n) ascending';
      break;
    case 'astar':
      preview.textContent = 'A*: f(n) = g(n) + h(n)';
      optBadge.textContent = 'Optimal (Admissible h)';
      optBadge.className = 'metric-badge optimal';
      openOrderHint.textContent = 'Sorted by f(n) = g + h';
      break;
  }
}

// --- 13. DOM EVENT REGISTRATION & INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize 20x20 Grid
  initializeGrid();

  // 2. Algorithm Dropdown Selector
  const algoSelect = document.getElementById('algorithm-select');
  algoSelect.addEventListener('change', (e) => {
    state.activeAlgorithm = e.target.value;
    updateFormulaPreview();
    resetPathOnly();
  });
  updateFormulaPreview();

  // 3. Speed Slider
  const speedSlider = document.getElementById('speed-slider');
  const speedLabel = document.getElementById('speed-label');
  function updateSpeedFromSlider() {
    // Slider 1 (slow, 120ms) to 100 (fast, 2ms)
    const val = parseInt(speedSlider.value, 10);
    // Invert scale: higher slider value = faster animation (lower ms)
    state.stepDelay = Math.max(2, Math.round(120 - (val * 1.15)));
    speedLabel.textContent = `${state.stepDelay} ms`;
  }
  speedSlider.addEventListener('input', updateSpeedFromSlider);
  updateSpeedFromSlider();

  // 4. Primary Buttons
  document.getElementById('btn-start').addEventListener('click', startOrResumeSearch);

  document.getElementById('btn-step').addEventListener('click', () => {
    if (state.isRunning) {
      stopSearchAnimation();
    }
    executeSingleStep();
  });

  document.getElementById('btn-reset-path').addEventListener('click', resetPathOnly);
  document.getElementById('btn-clear-walls').addEventListener('click', clearAllWalls);
  document.getElementById('btn-clear-all').addEventListener('click', clearEntireBoard);

  // 5. Maze Generation Buttons
  document.getElementById('btn-random-maze').addEventListener('click', () => {
    generateRandomObstacles();
  });
  document.getElementById('opt-maze-recursive').addEventListener('click', (e) => {
    e.stopPropagation();
    generateRecursiveMaze();
  });
  document.getElementById('opt-maze-random').addEventListener('click', (e) => {
    e.stopPropagation();
    generateRandomObstacles();
  });
  document.getElementById('opt-maze-spiral').addEventListener('click', (e) => {
    e.stopPropagation();
    generateSpiralMaze();
  });

  // 6. Drawing Tool Mode Selector Buttons
  document.querySelectorAll('.tool-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      setDrawingTool(btn.dataset.tool);
    });
  });

  // 7. OPEN vs CLOSED Sets Inspector Tabs
  document.querySelectorAll('.tab-btn').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.sets-panel').forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetPanel = document.getElementById(tab.dataset.target);
      if (targetPanel) targetPanel.classList.add('active');
    });
  });

  // 8. Theory Section Pills
  document.querySelectorAll('.theory-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.theory-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      renderTheorySection(pill.dataset.algo);
    });
  });
  // Initial theory render
  renderTheorySection('bfs');

  // 9. Comparison Modal Events
  document.getElementById('btn-compare').addEventListener('click', runComparisonBenchmark);
  document.getElementById('btn-close-modal').addEventListener('click', closeComparisonModal);
  document.getElementById('btn-modal-close').addEventListener('click', closeComparisonModal);
  document.getElementById('btn-modal-rerun').addEventListener('click', runComparisonBenchmark);

  // Close modal on backdrop click
  document.getElementById('compare-modal').addEventListener('click', (e) => {
    if (e.target.id === 'compare-modal') {
      closeComparisonModal();
    }
  });

  // Keyboard accessibility
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeComparisonModal();
    }
  });
});
