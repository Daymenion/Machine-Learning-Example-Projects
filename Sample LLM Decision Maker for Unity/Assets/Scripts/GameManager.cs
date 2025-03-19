using System.Collections.Generic;
using UnityEngine;



public class GameManager : MonoBehaviour
{
    [SerializeField] private GameObject tilePrefab;
    [SerializeField] private GameObject levelArea;
    [SerializeField] private GameObject passengerPrefab;
    [SerializeField] private GameObject busPrefab;
    [SerializeField] private GameObject waitingAreaPrefab;
    [SerializeField] private Color[] possibleColors;
    [SerializeField] private TextAsset[] levelDataFile;
    
    public Camera mainCamera;
    private float busDistance = 4f;
    private Tile[,] grid;
    private Bus currentBus;
    private Color nextBusColor;
    private int remainingPassengers;
    private int currentBusIndex;
    private LevelData currentLevel;
    private const float TileSpacing = 1.1f;
    private const int WaitingSlots = 5;
    private List<Vector2Int> currentPath;
    private WaitingSlot[] waitingSlots;
    private int currentLevelIndex;

    private void Start()
    {
        mainCamera = Camera.main;
       SetupLevel();
    }

    private void SetupLevel()
    {
        ClearLevel();
        LoadLevel();
        CreateGrid();
        CreateWaitingArea();
        SpawnNewBus();
        PositionCamera();
    }
    
    private Vector3 CalculateBusStartPosition()
    {
        float gridCenterX = (currentLevel.gridX - 1) * TileSpacing * 0.5f;
        float gridTopEdgeZ = busDistance;
        return new Vector3(gridCenterX, 0, gridTopEdgeZ);
    }

    private void ClearLevel()
    {
        foreach (Transform child in levelArea.transform)
        {
            Destroy(child.gameObject);
        }
    }

    private void LoadLevel()
    {
        currentLevel = JsonUtility.FromJson<LevelData>(levelDataFile[currentLevelIndex].text);
        currentBusIndex = 0;
        remainingPassengers = currentLevel.TotalPassengerCount();
    }

    private void LoadNextLevel()
    {
        currentLevelIndex++;
        if(currentLevelIndex >= levelDataFile.Length)
            currentLevelIndex = 0;
        SetupLevel();
    }

    private void CreateGrid()
    {
        grid = new Tile[currentLevel.gridX, currentLevel.gridY];

        
            for (int y = 0; y < currentLevel.gridY; y++)
            {
                for (int x = 0; x < currentLevel.gridX; x++)
                {
                    var cell = currentLevel.cellMatrix[y].cellArray[x];
                    Vector3 position = new Vector3(x * 1.1f, 0, y * -1.1f);
                    GameObject tileObj = Instantiate(tilePrefab, position, Quaternion.identity);
                    tileObj.transform.SetParent(levelArea.transform);
                    grid[x, y] = tileObj.GetComponent<Tile>();

                    grid[x, y].SetDisabled(cell.isDisabled);
                }
        }

        SpawnPassengers();
    }

    private void SpawnPassengers()
    {
        for(int y = 0; y < currentLevel.gridY; y++)
        {
            for(int x = 0; x < currentLevel.gridX; x++)
            {
                var cell = currentLevel.cellMatrix[y].cellArray[x];
                if (cell.passengers.Length > 0)
                {
                    var passengerData = cell.passengers[0];
                    Tile tile = grid[x, y];
                    GameObject passengerObj = Instantiate(passengerPrefab, tile.PassengerAnchor.position, Quaternion.identity);
                    Passenger passenger = passengerObj.GetComponent<Passenger>();
                    passenger.Initialize(possibleColors[passengerData.color]);
                    tile.SetPassenger(passenger);
                }
                
            }
           
        }
    }


    private void SpawnNewBus()
    {
        if (currentBusIndex >= currentLevel.busConfigs.Length)
            return;
        
        Vector3 busPosition = CalculateBusStartPosition();
        GameObject busObj = Instantiate(busPrefab, busPosition, Quaternion.identity);
        busObj.transform.SetParent(levelArea.transform);
        currentBus = busObj.GetComponent<Bus>();
        
        Color busColor = possibleColors[currentLevel.busConfigs[currentBusIndex].color];
        currentBus.Initialize(busColor);
        
        MoveMatchingPassengersFromWaiting();
    }
    
    private void CreateWaitingArea()
    {
        waitingSlots = new WaitingSlot[WaitingSlots];
        float startX = (currentLevel.gridX - WaitingSlots) * TileSpacing * 0.5f;
        float waitingAreaZ = 1 + TileSpacing;

        for (int i = 0; i < WaitingSlots; i++)
        {
            Vector3 position = new Vector3(startX + i * TileSpacing, 0, waitingAreaZ);
            GameObject slotObj = Instantiate(waitingAreaPrefab, position, Quaternion.identity);
            slotObj.transform.SetParent(levelArea.transform);
            waitingSlots[i] = new WaitingSlot { Position = position, Passenger = null};
        }
    }
    
    private int GetAvailableWaitingSlot()
    {
        for (int i = 0; i < waitingSlots.Length; i++)
        {
            if (waitingSlots[i].Passenger == null)
                return i;
        }
        return -1;
    }

    public void OnPassengerClicked(Passenger passenger)
    {
        if (currentPath != null) return;

        Vector2Int passengerPos = FindPassengerPosition(passenger);
        List<Vector2Int> path = FindPathToTop(passengerPos);
        
        if (path != null)
        {
            currentPath = path;
            if (passenger.Color == currentBus.Color)
            {
                StartCoroutine(MovePassengerToBus(passenger));
            }
            else
            {
                int slot = GetAvailableWaitingSlot();
                if (slot != -1)
                {
                    StartCoroutine(MovePassengerToWaitingArea(passenger, slot));
                }
            }
            MoveMatchingPassengersFromWaiting();
            
        }
    }
    
    private System.Collections.IEnumerator MovePassengerToWaitingArea(Passenger passenger, int slotIndex)
    {
        Vector2Int startPos = FindPassengerPosition(passenger);
        Tile startTile = grid[startPos.x, startPos.y];
        startTile.ClearPassenger();

        foreach (Vector2Int pathPos in currentPath)
        {
            yield return StartCoroutine(MovePassengerToPosition(passenger, 
                new Vector3(pathPos.x * TileSpacing, passenger.transform.position.y, -pathPos.y * TileSpacing)));
        }

        yield return StartCoroutine(MovePassengerToPosition(passenger, waitingSlots[slotIndex].Position));
        
        passenger.MakeNonPlayable();
        waitingSlots[slotIndex].Passenger = passenger;
        currentPath = null;

        if (GetAvailableWaitingSlot() == -1)
        {
            GameOver(false);
        }
    }
    
    private System.Collections.IEnumerator MovePassengerToBus(Passenger passenger)
    {
        Vector2Int startPos = FindPassengerPosition(passenger);
        Tile startTile = grid[startPos.x, startPos.y];
        startTile.ClearPassenger();

        foreach (Vector2Int pathPos in currentPath)
        {
            yield return StartCoroutine(MovePassengerToPosition(passenger, 
                new Vector3(pathPos.x * TileSpacing, passenger.transform.position.y, -pathPos.y * TileSpacing)));
        }

        currentBus.AddPassenger(passenger);
        passenger.MakeNonPlayable();
        
        remainingPassengers--;
        currentPath = null;

        CheckEnd();
    }

    public void CheckEnd()
    {
      
        if (currentBus.IsFull)
        {
            StartCoroutine(BusDepartureSequence());
          
        }

        if (remainingPassengers <= 0 || currentBusIndex >= currentLevel.busConfigs.Length)
        {
            GameOver(true);
        }
        
    }

    private System.Collections.IEnumerator MovePassengerToPosition(Passenger passenger, Vector3 targetPos)
    {
        float moveTime = 0.05f;
        float elapsed = 0;
        Vector3 startPos = passenger.transform.position;

        while (elapsed < moveTime)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / moveTime;
            passenger.transform.position = Vector3.Lerp(startPos, targetPos, t);
            yield return null;
        }
    }
    
    private void MoveMatchingPassengersFromWaiting()
    {
        for (int i = 0; i < waitingSlots.Length; i++)
        {
            if (waitingSlots[i].Passenger != null && 
                waitingSlots[i].Passenger.Color == currentBus.Color && 
                !currentBus.IsFull)
            {
                var passenger = waitingSlots[i].Passenger;
                waitingSlots[i].Passenger = null;
                StartCoroutine(MoveWaitingPassengerToBus(passenger));
            }
        }
    }

    private System.Collections.IEnumerator MoveWaitingPassengerToBus(Passenger passenger)
    {
        yield return StartCoroutine(MovePassengerToPosition(passenger, currentBus.transform.position));
        
        currentBus.AddPassenger(passenger);
        remainingPassengers--;

        CheckEnd();
    }


    private Vector2Int FindPassengerPosition(Passenger passenger)
    {
        for (int x = 0; x < currentLevel.gridX; x++)
        {
            for (int y = 0; y < currentLevel.gridY; y++)
            {
                if (grid[x, y].CurrentPassenger == passenger)
                {
                    return new Vector2Int(x, y);
                }
            }
        }
        return Vector2Int.zero;
    }

    private List<Vector2Int> FindPathToTop(Vector2Int start)
    {
        var visited = new HashSet<Vector2Int>();
        var queue = new Queue<Vector2Int>();
        var cameFrom = new Dictionary<Vector2Int, Vector2Int>();
        
        queue.Enqueue(start);
        visited.Add(start);
        
        while (queue.Count > 0)
        {
            Vector2Int current = queue.Dequeue();
            
            if (current.y == 0) // Reached top row
            {
                return ReconstructPath(start, current, cameFrom);
            }
            
            foreach (Vector2Int next in GetValidMoves(current))
            {
                if (!visited.Contains(next))
                {
                    visited.Add(next);
                    queue.Enqueue(next);
                    cameFrom[next] = current;
                }
            }
        }
        
        return null;
    }

    private List<Vector2Int> GetValidMoves(Vector2Int pos)
    {
        var moves = new List<Vector2Int>();
        Vector2Int[] directions = { Vector2Int.up, Vector2Int.down, Vector2Int.left, Vector2Int.right };
        
        foreach (Vector2Int dir in directions)
        {
            Vector2Int next = pos + dir;
            if (IsValidPosition(next) && grid[next.x, next.y].CurrentPassenger == null && !grid[next.x, next.y].IsDisabled)
            {
                moves.Add(next);
            }
        }
        
        return moves;
    }

    private bool IsValidPosition(Vector2Int pos)
    {
        return pos.x >= 0 && pos.x < currentLevel.gridX && 
               pos.y >= 0 && pos.y < currentLevel.gridY;
    }

    private List<Vector2Int> ReconstructPath(Vector2Int start, Vector2Int end, Dictionary<Vector2Int, Vector2Int> cameFrom)
    {
        var path = new List<Vector2Int>();
        Vector2Int current = end;
        
        while (!current.Equals(start))
        {
            path.Add(current);
            current = cameFrom[current];
        }
        path.Add(start);
        path.Reverse();
        return path;
    }

    

    private System.Collections.IEnumerator BusDepartureSequence()
    {
        yield return new WaitForSeconds(0.2f);
        
        Vector3 startPos = CalculateBusStartPosition();
        Vector3 exitPosition = new Vector3(startPos.x + 10f, 0, startPos.z);
        float duration = 0.5f;
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / duration;
            currentBus.transform.position = Vector3.Lerp(startPos, exitPosition, t);
            yield return null;
        }
        
        Destroy(currentBus.gameObject);
        
        currentBusIndex++;
        SpawnNewBus();
    
    }
    
    private void PositionCamera()
    {
        // yield return new WaitForEndOfFrame();
        FocusTarget(levelArea);
    }
    
    private Bounds CalculateBounds(GameObject target)
    {
        Renderer[] renderers = target.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0) return new Bounds();

        Bounds combinedBounds = renderers[0].bounds;
        foreach (Renderer renderer in renderers)
        {
            combinedBounds.Encapsulate(renderer.bounds);
        }

        return combinedBounds;
    }
    
    public void FocusTarget(GameObject target)
    {
        if (target == null || mainCamera == null) return;

        Bounds bounds = CalculateBounds(target);
        if (bounds.size == Vector3.zero) return;
        
        float objectSize = Mathf.Max(bounds.size.x, bounds.size.y, bounds.size.z);
        float distance = objectSize / (2.0f * Mathf.Tan(Mathf.Deg2Rad * mainCamera.fieldOfView / 2.0f));
        distance *= 1.4f; // Add padding to the distance

        //Vector3 direction = (mainCamera.transform.position - bounds.center).normalized;
        Vector3 targetPosition = bounds.center + levelArea.transform.up * distance;


        StartCoroutine(SmoothMoveCamera(mainCamera.transform.position, targetPosition, 0.01f));
    }
    
    private System.Collections.IEnumerator SmoothMoveCamera(Vector3 startPos, Vector3 endPos, float duration)
    {
        float elapsedTime = 0f;
        Vector3 velocity = Vector3.zero;
        while (elapsedTime < duration)
        {
            mainCamera.transform.position = Vector3.SmoothDamp(startPos, endPos, ref velocity, duration);
            elapsedTime += Time.deltaTime;
            yield return null;
        }

        // Finalize position
        mainCamera.transform.position = endPos;
    }

    private void GameOver(bool won)
    {
        StartCoroutine(GameOverSequence(won));
    }

    private System.Collections.IEnumerator GameOverSequence(bool won)
    {
        yield return new WaitForSeconds(1f);
        
        Debug.Log(won ? "Level Complete!" : "Level Failed - Waiting Area Full!");

        if (won)
        {
            LoadNextLevel();
        }
        else
        {
            SetupLevel();
        }
    }
}