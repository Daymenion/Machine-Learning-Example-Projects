using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
// Remove System.Text.Json reference as it's not available in Unity's .NET Framework 4.7.1

public class AIAgent : MonoBehaviour
{
    [SerializeField] private GameManager gameManager;
    [SerializeField] private string apiKey;
    [SerializeField] private float thinkingDelay = 1.0f;
    [SerializeField] private bool activateAI = true;
    [SerializeField] private string model = "gpt-4o-mini"; // Or "gpt-4o" for better reasoning
    [SerializeField] private bool enableLogging = true;
    [SerializeField, Range(0.0f, 1.0f)] private float temperature = 0.3f;
    [SerializeField] private bool waitForBusDeparture = true;

    private bool isThinking = false;
    private bool isWaitingForBusDeparture = false;
    private Tile[,] grid;
    private WaitingSlot[] waitingSlots;
    private Bus currentBus;
    private int currentLevelIndex;
    private int moveCount = 0;
    private int levelSuccessCount = 0;
    private StreamWriter logWriter;
    private string logFilePath;
    private List<string> gameHistory = new List<string>();
    
    [Serializable]
    private class Message
    {
        public string role;
        public string content;
    }

    [Serializable]
    private class OpenAIRequest
    {
        public string model;
        public List<Message> messages;
        public float temperature = 0.3f;
    }

    [Serializable]
    private class OpenAIChoice
    {
        public Message message = new Message();

        // Default constructor to prevent null field
        public OpenAIChoice() { }
    }

    [Serializable]
    private class OpenAIResponse
    {
        public List<OpenAIChoice> choices = new List<OpenAIChoice>();

        // Default constructor to prevent null field
        public OpenAIResponse() { }
    }

    private void Awake()
    {
        if (enableLogging)
        {
            InitializeLogging();
        }
    }

    private void InitializeLogging()
    {
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string directory = Path.Combine(Application.persistentDataPath, "AILogs");
        
        if (!Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }
        
        logFilePath = Path.Combine(directory, $"busJamAI_{timestamp}.log");
        logWriter = new StreamWriter(logFilePath, true);
        logWriter.WriteLine("=== Bus Jam AI Agent Log ===");
        logWriter.WriteLine($"Session started: {DateTime.Now}");
        logWriter.WriteLine($"Model: {model}, Temperature: {temperature}");
        logWriter.WriteLine("===========================");
        logWriter.Flush();
        
        Debug.Log($"AI Agent logging to: {logFilePath}");
    }

    private void OnDestroy()
    {
        if (logWriter != null)
        {
            logWriter.WriteLine($"Session ended: {DateTime.Now}");
            logWriter.WriteLine($"Total levels completed: {levelSuccessCount}");
            logWriter.Close();
        }
    }

    private void Start()
    {
        if (!ValidateSetup())
        {
            enabled = false;
            return;
        }

        // Initialize game state immediately
        CaptureGameState();
        
        if (activateAI)
        {
            StartCoroutine(AIGameplayLoop());
        }
    }

    private bool ValidateSetup()
    {
        bool isValid = true;
        
        // Check GameManager reference
        if (gameManager == null)
        {
            Debug.LogError("AIAgent setup error: GameManager reference is missing. Please assign the GameManager object in the Inspector.");
            isValid = false;
        }
        
        // Check API Key
        if (string.IsNullOrEmpty(apiKey))
        {
            Debug.LogError("AIAgent setup error: API Key is empty. Please add your OpenAI API key in the Inspector.");
            isValid = false;
        }
        else if (apiKey == "YOUR_API_KEY" || apiKey == "sk-...")
        {
            Debug.LogWarning("AIAgent setup warning: API Key appears to be a placeholder. Make sure to use a valid OpenAI API key.");
        }
        
        // Log successful setup
        if (isValid)
        {
            Debug.Log("AIAgent setup validated successfully!");
            Debug.Log($"Using model: {model}, Temperature: {temperature}, Thinking delay: {thinkingDelay}s");
            Debug.Log("AI Agent is ready to play. Make sure the game is started before the AI begins making decisions.");
        }
        
        return isValid;
    }

    private IEnumerator AIGameplayLoop()
    {
        // Wait for the first level to be fully set up
        yield return new WaitForSeconds(2.0f);
        
        while (true)
        {
            // If the bus is full, we need to wait for it to leave before making another decision
            if (isWaitingForBusDeparture)
            {
                yield return new WaitForSeconds(1f);
                CaptureGameState();
                
                // Check if a new bus has arrived
                if (currentBus != null && !currentBus.IsFull)
                {
                    isWaitingForBusDeparture = false;
                    LogAction("Bus departed, continuing gameplay.");
                }
                else
                {
                    continue;
                }
            }
            
            if (!isThinking)
            {
                isThinking = true;
                CaptureGameState();
                
                // Check if we've moved to a new level
                int detectedLevel = GetCurrentLevelIndex();
                if (detectedLevel != currentLevelIndex)
                {
                    currentLevelIndex = detectedLevel;
                    levelSuccessCount++;
                    moveCount = 0;
                    LogAction($"Level {currentLevelIndex} started!");
                    gameHistory.Clear();
                }
                
                // Handle the case where the bus is full
                if (currentBus != null && currentBus.IsFull && waitForBusDeparture)
                {
                    LogAction("Bus is full, waiting for departure...");
                    isWaitingForBusDeparture = true;
                    isThinking = false;
                    continue;
                }
                
                StartCoroutine(MakeAIDecision());
            }
            
            yield return new WaitForSeconds(thinkingDelay);
        }
    }

    private int GetCurrentLevelIndex()
    {
        try
        {
            return (int)typeof(GameManager)
                .GetField("currentLevelIndex", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                .GetValue(gameManager);
        }
        catch
        {
            return 0;
        }
    }

    private void CaptureGameState()
    {
        // Get references to game state through reflection
        try
        {
            grid = (Tile[,])typeof(GameManager)
                .GetField("grid", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                .GetValue(gameManager);
                
            waitingSlots = (WaitingSlot[])typeof(GameManager)
                .GetField("waitingSlots", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                .GetValue(gameManager);
                
            currentBus = (Bus)typeof(GameManager)
                .GetField("currentBus", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                .GetValue(gameManager);
                
            // Handle cases where these might be null (between levels)
            if (grid == null || waitingSlots == null || currentBus == null)
            {
                LogAction("Game state not fully initialized yet.");
                isThinking = false;
            }
        }
        catch (System.Exception ex)
        {
            LogAction($"Error capturing game state: {ex.Message}");
            isThinking = false;
        }
    }

    private string ColorToString(Color color)
    {
        // Support all 10 colors found in the game
        // Blue
        if (Mathf.Abs(color.r - 0.05f) < 0.1f && Mathf.Abs(color.g - 0.22f) < 0.1f && Mathf.Abs(color.b - 0.75f) < 0.1f) return "Blue";
        // Purple
        if (Mathf.Abs(color.r - 0.75f) < 0.1f && Mathf.Abs(color.g - 0.05f) < 0.1f && Mathf.Abs(color.b - 0.74f) < 0.1f) return "Purple";
        // Red
        if (Mathf.Abs(color.r - 0.75f) < 0.1f && Mathf.Abs(color.g - 0.05f) < 0.1f && Mathf.Abs(color.b - 0.05f) < 0.1f) return "Red";
        // Green
        if (Mathf.Abs(color.r - 0.08f) < 0.1f && Mathf.Abs(color.g - 0.75f) < 0.1f && Mathf.Abs(color.b - 0.05f) < 0.1f) return "Green";
        // Yellow
        if (Mathf.Abs(color.r - 0.97f) < 0.1f && Mathf.Abs(color.g - 1.00f) < 0.1f && Mathf.Abs(color.b - 0.00f) < 0.1f) return "Yellow";
        // Orange
        if (Mathf.Abs(color.r - 0.97f) < 0.1f && Mathf.Abs(color.g - 0.62f) < 0.1f && Mathf.Abs(color.b - 0.14f) < 0.1f) return "Orange";
        // White
        if (Mathf.Abs(color.r - 1.00f) < 0.1f && Mathf.Abs(color.g - 1.00f) < 0.1f && Mathf.Abs(color.b - 1.00f) < 0.1f) return "White";
        // Black
        if (Mathf.Abs(color.r - 0.00f) < 0.1f && Mathf.Abs(color.g - 0.00f) < 0.1f && Mathf.Abs(color.b - 0.00f) < 0.1f) return "Black";
        // Teal
        if (Mathf.Abs(color.r - 0.06f) < 0.1f && Mathf.Abs(color.g - 0.80f) < 0.1f && Mathf.Abs(color.b - 0.66f) < 0.1f) return "Teal";
        // Pink
        if (Mathf.Abs(color.r - 0.86f) < 0.1f && Mathf.Abs(color.g - 0.58f) < 0.1f && Mathf.Abs(color.b - 0.75f) < 0.1f) return "Pink";
        
        // Fallback for general color ranges (less precise)
        if (color.r > 0.7f && color.g < 0.3f && color.b < 0.3f) return "Red";
        if (color.r < 0.3f && color.g > 0.7f && color.b < 0.3f) return "Green";
        if (color.r < 0.3f && color.g < 0.3f && color.b > 0.7f) return "Blue";
        if (color.r > 0.7f && color.g > 0.7f && color.b < 0.3f) return "Yellow";
        return "Unknown";
    }

    private string ColorToCode(Color color)
    {
        // Get the color name and return a unique code for each
        string colorName = ColorToString(color);
        
        switch (colorName)
        {
            case "Red": return "R";
            case "Green": return "G";
            case "Blue": return "B";
            case "Yellow": return "Y";
            case "Purple": return "P";
            case "Orange": return "O";
            case "White": return "W";
            case "Black": return "K";  // Using K for black
            case "Teal": return "T";
            case "Pink": return "N";   // Using N for pink
            default: return "?";
        }
    }

    private IEnumerator MakeAIDecision()
    {
        if (string.IsNullOrEmpty(apiKey))
        {
            LogAction("API key is not set. Cannot make AI decision.");
            yield break;
        }
        
        // Declare jsonRequest here so it's available throughout the method
        string jsonRequest = "";
        
        try
        {
            isThinking = true;
            
            // Build the current game state as a prompt
            string gameState = BuildGameStatePrompt();
            LogGameState(gameState);
            
            // Prepare a crystal clear system message focused on MOVE COST
            string systemMessage = 
                "You are playing Bus Jam, a puzzle game where you need to optimize moves to get matching-color passengers to buses.\n\n" +
                "KEY RULES (FOLLOW PRECISELY):\n\n" +
                "1) ALWAYS move passengers with CLEAR PATHS to the bus first. These have a move cost of 1.\n\n" +
                "2) If no passenger has a clear path, move the blocker of the passenger with the LOWEST MOVE COST.\n\n" +
                "3) Move cost = (number of blockers + 1). Always minimize the total move cost.\n\n" +
                "4) Coordinates are [x,y] where x is HORIZONTAL and y is VERTICAL.\n\n" +
                "5) The MATCHING PASSENGER ANALYSIS section shows the exact move costs - trust this analysis.\n\n" +
                "6) RESPOND WITH COORDINATES IN THE FORMAT [1,2] WITHOUT ANY PREFIXES (no 'x=' or 'y=').";
            
            // Create the message structure
            List<Message> messages = new List<Message>
            {
                new Message { role = "system", content = systemMessage },
                new Message { role = "user", content = gameState + "\n\nWhat is your next move?\nRespond with ONLY the coordinates [x,y] of the passenger you want to move, making sure x is HORIZONTAL and y is VERTICAL.\nIMPORTANT: Format your answer as simple coordinates like [1,2] WITHOUT using 'x=' and 'y=' prefixes. Then briefly explain why this is the optimal move." }
            };
            
            // Create the request object
            OpenAIRequest request = new OpenAIRequest
            {
                model = model,
                messages = messages,
                temperature = temperature
            };
            
            // Convert to JSON
            jsonRequest = JsonUtility.ToJson(request);
        }
        catch (Exception e)
        {
            LogAction("Error preparing AI request: " + e.Message);
            isThinking = false;
            yield break;
        }
        
        // Wait for the thinking delay to make it feel more natural
        yield return new WaitForSeconds(thinkingDelay);
        
        // Create the web request
        UnityWebRequest www = null;
        
        try
        {
            // Send the API request
            www = new UnityWebRequest("https://api.openai.com/v1/chat/completions", "POST");
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonRequest);
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");
            www.SetRequestHeader("Authorization", "Bearer " + apiKey);
            
            LogAction("Sending API request...");
        }
        catch (Exception e)
        {
            LogAction("Error setting up web request: " + e.Message);
            isThinking = false;
            
            if (www != null)
            {
                www.Dispose();
            }
            
            yield break;
        }
        
        // Send the request (outside the try block)
        yield return www.SendWebRequest();
        
        try
        {
            if (www.result != UnityWebRequest.Result.Success)
            {
                LogAction("API Error: " + www.error);
                isThinking = false;
                www.Dispose();
                yield break;
            }
            
            string responseJson = www.downloadHandler.text;
            LogAction("API Response received (" + responseJson.Length + " chars)");
            LogAction("Raw response: " + responseJson);
            
            string aiDecision = ExtractMessageContent(responseJson);
            LogAction("AI Decision: " + aiDecision);
            
            // Parse and execute a single move
            ParseAndExecuteMove(aiDecision);
        }
        catch (Exception e)
        {
            LogAction("Error processing API response: " + e.Message);
        }
        finally
        {
            if (www != null)
            {
                www.Dispose();
            }
            
            isThinking = false;
        }
    }

    // Method to parse and execute a single move
    private void ParseAndExecuteMove(string aiDecision)
    {
        try
        {
            // Extract [x,y] coordinates from the AI's response
            int indexOfOpenBracket = aiDecision.IndexOf('[');
            int indexOfCloseBracket = aiDecision.IndexOf(']');
            
            if (indexOfOpenBracket >= 0 && indexOfCloseBracket > indexOfOpenBracket)
            {
                string coordinatesStr = aiDecision.Substring(
                    indexOfOpenBracket + 1, 
                    indexOfCloseBracket - indexOfOpenBracket - 1);
                
                string[] coordinates = coordinatesStr.Split(',');
                
                // Handle both formats: [1,2] and [x=1,y=2]
                int x = -1, y = -1;
                
                if (coordinates.Length == 2)
                {
                    string xStr = coordinates[0].Trim();
                    string yStr = coordinates[1].Trim();
                    
                    // Handle format with x= and y= prefixes
                    if (xStr.StartsWith("x="))
                    {
                        xStr = xStr.Substring(2); // Remove "x=" prefix
                    }
                    
                    if (yStr.StartsWith("y="))
                    {
                        yStr = yStr.Substring(2); // Remove "y=" prefix
                    }
                    
                    // Parse the values
                    bool validX = int.TryParse(xStr, out x);
                    bool validY = int.TryParse(yStr, out y);
                    
                    if (validX && validY)
                    {
                        string moveInfo = $"Move {++moveCount}: Selected passenger at [{x},{y}]";
                        LogAction(moveInfo);
                        gameHistory.Add(moveInfo);
                        
                        // First check if the coordinates are within grid bounds
                        bool foundPassenger = false;
                        
                        // Check if original coordinates are valid
                        if (x >= 0 && x < grid.GetLength(0) && y >= 0 && y < grid.GetLength(1))
                        {
                            Tile targetTile = grid[x, y];
                            if (targetTile != null && targetTile.CurrentPassenger != null)
                            {
                                // We found a valid passenger, so continue with the original logic
                                foundPassenger = true;
                                ProcessPassengerMove(x, y, targetTile.CurrentPassenger);
                            }
                            else
                            {
                                LogAction($"No passenger found at [{x},{y}], will try checking for coordinate swap");
                            }
                        }
                        else
                        {
                            LogAction($"Coordinates [{x},{y}] are out of bounds, will try checking for coordinate swap");
                        }
                        
                        // If no passenger was found, check if the coordinates are swapped
                        if (!foundPassenger)
                        {
                            // Try the swapped coordinates
                            int swappedX = y;
                            int swappedY = x;
                            
                            // Check if swapped coordinates are valid
                            if (swappedX >= 0 && swappedX < grid.GetLength(0) && swappedY >= 0 && swappedY < grid.GetLength(1))
                            {
                                Tile swappedTargetTile = grid[swappedX, swappedY];
                                if (swappedTargetTile != null && swappedTargetTile.CurrentPassenger != null)
                                {
                                    // We found a passenger at the swapped coordinates
                                    LogAction($"Found passenger at swapped coordinates [{swappedX},{swappedY}], likely coordinate confusion");
                                    ProcessPassengerMove(swappedX, swappedY, swappedTargetTile.CurrentPassenger);
                                    foundPassenger = true;
                                }
                            }
                            
                            if (!foundPassenger)
                            {
                                LogAction($"No valid passenger found at original [{x},{y}] or swapped [{swappedX},{swappedY}] coordinates");
                            }
                        }
                    }
                    else
                    {
                        LogAction($"Invalid coordinate values: x={xStr}, y={yStr}");
                    }
                }
                else
                {
                    LogAction($"Invalid coordinates format: {coordinatesStr}");
                }
            }
            else
            {
                LogAction("Couldn't find coordinates in AI response: " + aiDecision);
            }
        }
        catch (Exception e)
        {
            LogAction("Error parsing move: " + e.Message);
        }
    }
    
    // Helper method to process a passenger move
    private void ProcessPassengerMove(int x, int y, Passenger passenger)
    {
        // Same coordinate system as the grid display
        if (passenger != null)
        {
            // Get color name for logging
            string colorName = ColorToString(passenger.Color); // Replace IdentifyColor with ColorToString
            string busFull = currentBus != null && currentBus.IsFull ? "true" : "false";
            
            LogAction($"Clicking {colorName} passenger at [{x},{y}]. Current bus: {(currentBus != null ? ColorToString(currentBus.Color) : "None")}, full: {busFull}");
            
            // Check if this passenger matches the current bus
            bool isMatch = currentBus != null && IsColorMatch(passenger.Color, currentBus.Color);
            if (isMatch)
            {
                LogAction($"This passenger matches the current bus color ({ColorToString(currentBus.Color)})");
            }
            else
            {
                LogAction($"This passenger does not match the current bus color");
            }
            
            // Prevent clicking if the matching bus is already full
            if (isMatch && currentBus != null && currentBus.IsFull)
            {
                LogAction("Cannot move this passenger - the matching bus is already full");
                return;
            }
            
            // Forward the click to GameManager
            gameManager.OnPassengerClicked(passenger);
        }
        else
        {
            LogAction($"No passenger found at [{x},{y}]");
        }
    }

    private bool IsColorMatch(Color c1, Color c2)
    {
        // Simple direct color comparison like GameManager.cs does
        bool isMatch = c1 == c2;
        
        // Log for debugging
        if (isMatch)
        {
            LogAction($"Color comparison: {ColorToString(c1)} (R={c1.r:F2}, G={c1.g:F2}, B={c1.b:F2}) vs {ColorToString(c2)} (R={c2.r:F2}, G={c2.g:F2}, B={c2.b:F2}), Match: {isMatch}");
        }
        
        return isMatch;
    }
    
    private void LogAction(string message)
    {
        if (!enableLogging || logWriter == null) return;
        
        string timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
        string logEntry = $"[{timestamp}] {message}";
        
        Debug.Log(logEntry);
        logWriter.WriteLine(logEntry);
        logWriter.Flush();
    }
    
    private void LogGameState(string gameState)
    {
        if (!enableLogging || logWriter == null) return;
        
        string timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
        logWriter.WriteLine($"[{timestamp}] === GAME STATE ===");
        logWriter.WriteLine(gameState);
        logWriter.WriteLine($"[{timestamp}] === END GAME STATE ===");
        logWriter.Flush();
    }

    private string ExtractMessageContent(string responseJson)
    {
        try
        {
            // Look for "content" field in the response
            int contentIndex = responseJson.IndexOf("\"content\":");
            if (contentIndex < 0)
            {
                LogAction("No content field found in response");
                return null;
            }
            
            // Find where the actual content string starts (after the quotes)
            int startQuote = responseJson.IndexOf('"', contentIndex + 10); // "content": plus some space
            if (startQuote < 0)
            {
                LogAction("No opening quote for content found");
                return null;
            }
            
            // Now find the closing quote, but we need to handle escaped quotes
            int endQuote = -1;
            bool inEscape = false;
            for (int i = startQuote + 1; i < responseJson.Length; i++)
            {
                if (inEscape)
                {
                    inEscape = false;
                    continue;
                }
                
                if (responseJson[i] == '\\')
                {
                    inEscape = true;
                    continue;
                }
                
                if (responseJson[i] == '"')
                {
                    endQuote = i;
                    break;
                }
            }
            
            if (endQuote < 0)
            {
                LogAction("No closing quote for content found");
                return null;
            }
            
            // Extract the content
            string content = responseJson.Substring(startQuote + 1, endQuote - startQuote - 1);
            
            // Replace escaped characters
            content = content.Replace("\\\"", "\"")
                             .Replace("\\n", "\n")
                             .Replace("\\r", "\r")
                             .Replace("\\t", "\t")
                             .Replace("\\\\", "\\");
            
            return content;
        }
        catch (Exception ex)
        {
            LogAction($"Error in ExtractMessageContent: {ex.Message}");
            return null;
        }
    }

    private string BuildGameStatePrompt()
    {
        // Debug logging of all colors in the game state
        LogColorsInGameState();
        
        StringBuilder sb = new StringBuilder();
        
        // Add game rules and explanation
        sb.AppendLine("======= BUS JAM GAME STATE =======");
        sb.AppendLine();
        sb.AppendLine("GAME RULES:");
        sb.AppendLine("1. Move matching-color passengers to the bus when they can reach the top row");
        sb.AppendLine("2. Move non-matching passengers to the waiting area only if they block a matching passenger");
        sb.AppendLine("3. You lose if the waiting area fills up, so use it sparingly");
        sb.AppendLine("4. Always choose the matching passenger that requires the FEWEST moves to reach the bus");
        sb.AppendLine();
        
        // Add clear note about coordinate system
        sb.AppendLine("COORDINATES: [x,y] where x=HORIZONTAL, y=VERTICAL");
        sb.AppendLine("For example, [12,16] means the passenger in column 12, row 16");
        sb.AppendLine();
        
        // Add current level and move count
        sb.AppendLine($"Current level: {GetCurrentLevelIndex()}");
        sb.AppendLine($"Move count: {moveCount}");
        sb.AppendLine();
        
        // Add current bus information
        if (currentBus != null)
        {
            string busColorName = ColorToString(currentBus.Color);
            sb.AppendLine($"Current bus color: {busColorName}");
            sb.AppendLine($"Is bus full: {currentBus.IsFull}");
            sb.AppendLine();
        }
        
        // Add waiting area status
        if (waitingSlots != null)
        {
            sb.AppendLine("Waiting area status:");
            
            int filledCount = 0;
            for (int i = 0; i < waitingSlots.Length; i++)
            {
                if (waitingSlots[i].Passenger != null)
                {
                    string passengerColor = ColorToString(waitingSlots[i].Passenger.Color);
                    sb.AppendLine($"Slot {i+1}: {passengerColor} passenger");
                    filledCount++;
                }
                else
                {
                    sb.AppendLine($"Slot {i+1}: Empty");
                }
            }
            
            sb.AppendLine($"Waiting area: {filledCount}/{waitingSlots.Length} slots filled");
            sb.AppendLine();
        }
        
        // Add grid state
        sb.AppendLine("GRID STATE:");
        int gridSizeX = grid.GetLength(0);
        int gridSizeY = grid.GetLength(1);
        
        for (int y = 0; y < gridSizeY; y++)
        {
            StringBuilder row = new StringBuilder();
            row.Append($"Row {y}: ");
            
            for (int x = 0; x < gridSizeX; x++)
            {
                Tile tile = grid[x, y];
                if (tile == null || tile.IsDisabled)
                {
                    row.Append("X ");  // Disabled or null tile
                }
                else if (tile.CurrentPassenger == null)
                {
                    row.Append(". ");  // Empty tile
                }
                else
                {
                    row.Append(ColorToCode(tile.CurrentPassenger.Color) + " ");  // Passenger with color code
                }
            }
            
            sb.AppendLine(row.ToString());
        }
        
        sb.AppendLine();
        sb.AppendLine("Legend: . = Empty, X = Disabled, R = Red, B = Blue, G = Green, Y = Yellow, P = Purple");
        sb.AppendLine();
        
        // Add path analysis with move cost calculation - ONLY for matching color passengers
        sb.AppendLine("======= MATCHING PASSENGER ANALYSIS =======");
        
        if (currentBus != null && grid != null)
        {
            Color busColor = currentBus.Color;
            string busColorName = ColorToString(busColor);
            bool foundMatchingPassengers = false;
            
            // First pass to check if there are any matching passengers at all
            List<Vector2Int> matchingPassengers = new List<Vector2Int>();
            
            for (int y = 0; y < gridSizeY; y++)
            {
                for (int x = 0; x < gridSizeX; x++)
                {
                    Tile tile = grid[x, y];
                    if (tile != null && !tile.IsDisabled && tile.CurrentPassenger != null)
                    {
                        if (IsColorMatch(tile.CurrentPassenger.Color, busColor))
                        {
                            matchingPassengers.Add(new Vector2Int(x, y));
                            foundMatchingPassengers = true;
                        }
                    }
                }
            }
            
            if (!foundMatchingPassengers)
            {
                sb.AppendLine($"No {busColorName} passengers found on the grid.");
                sb.AppendLine();
            }
            else
            {
                sb.AppendLine($"Found {matchingPassengers.Count} {busColorName} passengers that match the current bus.");
                sb.AppendLine();
                
                
                Dictionary<Vector2Int, int> positionToMoveCost = new Dictionary<Vector2Int, int>();
                
                foreach (Vector2Int pos in matchingPassengers)
                {
                    int x = pos.x;
                    int y = pos.y;
                    
                    // First check if passenger has a clear path
                    if (HasClearPathToTop(x, y))
                    {
                        sb.AppendLine($"★★★ PASSENGER AT [{x},{y}] HAS A CLEAR PATH TO THE BUS! ★★★");
                        sb.AppendLine($"This passenger has a clear path (direct move to bus)");
                        positionToMoveCost[pos] = 1;
                    }
                    else
                    {
                        // If no clear path, calculate how many blockers need to be moved
                        List<string> blockers = FindBlockingPositions(x, y);
                        int moveCost = blockers.Count + 1; // +1 for the final move to the bus
                        
                        sb.AppendLine($"Passenger at [{x},{y}] has {blockers.Count} blockers.");
                        sb.AppendLine($"Move cost: {moveCost} ({blockers.Count} blockers + 1 final move)");
                        sb.AppendLine($"Blockers: {string.Join(", ", blockers)}");
                        
                        sb.AppendLine($"BEST BLOCKER TO MOVE FOR THIS PASSENGER: {blockers[0]}");
                        
                        
                        positionToMoveCost[pos] = moveCost;
                    }
                }
            }
        }
        
        sb.AppendLine();
        sb.AppendLine("WHAT TO DO:");
        sb.AppendLine("1. If there's a passenger with a clear path to the bus, MOVE IT");
        sb.AppendLine("2. Otherwise, move the FIRST blocker of the passenger with the LOWEST move cost");
        
        return sb.ToString();
    }

    // Path-finding helper methods
    private bool HasClearPathToTop(int x, int y)
    {
        // If already at the top row, it has a clear path
        if (y == 0) 
        {
            LogAction($"CLEAR PATH: Passenger at [{x},{y}] is already at the top row");
            return true;
        }
        
        int gridWidth = grid.GetLength(0);
        int gridHeight = grid.GetLength(1);
        
        // Create a simplified grid representation marking occupied spaces
        bool[,] occupied = new bool[gridWidth, gridHeight];
        
        // Mark all spaces with passengers or disabled tiles
        for (int gY = 0; gY < gridHeight; gY++)
        {
            for (int gX = 0; gX < gridWidth; gX++)
            {
                Tile tile = grid[gX, gY];
                if (tile == null || tile.IsDisabled || (tile.CurrentPassenger != null && (gX != x || gY != y)))
                {
                    occupied[gX, gY] = true;
                }
                else
                {
                    occupied[gX, gY] = false;
                }
            }
        }
        
        LogAction("Exploring paths to top row...");
        // OPTIMIZATION #2: If no direct path, check horizontal+vertical paths
        if (HasAnyPathToTop(x, y, occupied))
        {
            return true;
        }
        
        // No path found
        LogAction($"NO CLEAR PATH: Passenger at [{x},{y}] cannot reach the top row");
        return false;
    }

    
    // Helper method to check if there's any path to the top using BFS
    private bool HasAnyPathToTop(int x, int y, bool[,] occupied)
    {
        int gridWidth = occupied.GetLength(0);
        int gridHeight = occupied.GetLength(1);
        
        // Use breadth-first search (BFS) to find any path to the top row
        Queue<Vector2Int> queue = new Queue<Vector2Int>();
        bool[,] visited = new bool[gridWidth, gridHeight];
        
        // Start from the passenger's position
        queue.Enqueue(new Vector2Int(x, y));
        visited[x, y] = true;
        
        // Direction vectors for Up, Right, Down, Left
        int[] dx = { 0, 1, 0, -1 };
        int[] dy = { -1, 0, 1, 0 };
        

        
        while (queue.Count > 0)
        {
            Vector2Int current = queue.Dequeue();
            
            // If we've reached the top row, there's a path
            if (current.y == 0)
            {
                LogAction($"CLEAR PATH: Passenger at [{x},{y}] can reach top row at position [{current.x},0]");
                return true;
            }
            
            // Try all four directions
            for (int i = 0; i < 4; i++)
            {
                int nx = current.x + dx[i];
                int ny = current.y + dy[i];
                
                // Check if this position is valid and not visited
                if (nx >= 0 && nx < gridWidth && ny >= 0 && ny < gridHeight && 
                    !visited[nx, ny] && !occupied[nx, ny])
                {
                    queue.Enqueue(new Vector2Int(nx, ny));
                    visited[nx, ny] = true;
                }
            }
        }
        
        // If we get here, no path was found
        return false;
    }

    private List<string> FindBlockingPositions(int x, int y)
    {
        // Quick return if passenger already has a clear path
        if (HasClearPathToTop(x, y))
        {
            LogAction($"Passenger at [{x},{y}] has a clear path - no blockers needed");
            return new List<string>();
        }
        
        int gridWidth = grid.GetLength(0);
        int gridHeight = grid.GetLength(1);
        
        // Create occupied grid
        bool[,] occupied = new bool[gridWidth, gridHeight];
        for (int gY = 0; gY < gridHeight; gY++)
        {
            for (int gX = 0; gX < gridWidth; gX++)
            {
                Tile tile = grid[gX, gY];
                occupied[gX, gY] = tile == null || tile.IsDisabled || 
                                  (tile.CurrentPassenger != null && (gX != x || gY != y));
            }
        }
        
        LogAction($"Finding optimal path with minimal blockers for passenger at [{x},{y}]");
        
        // Keep track of paths by blocker count
        Dictionary<int, List<Vector2Int>> pathsByBlockerCount = new Dictionary<int, List<Vector2Int>>();
        
        // APPROACH 1: Check if we can create a path by removing just ONE blocker
        // Try removing each passenger to see if it creates a path by itself
        for (int gY = 0; gY < gridHeight; gY++)
        {
            for (int gX = 0; gX < gridWidth; gX++)
            {
                // Skip checking our passenger or non-passenger tiles
                if ((gX == x && gY == y) || 
                    grid[gX, gY] == null || 
                    grid[gX, gY].IsDisabled || 
                    grid[gX, gY].CurrentPassenger == null)
                {
                    continue;
                }
                
                // Try removing this passenger
                bool[,] tempGrid = (bool[,])occupied.Clone();
                tempGrid[gX, gY] = false;
                
                // Check if this creates a path
                if (HasAnyPathToTop(x, y, tempGrid))
                {
                    // We found a single-blocker solution!
                    if (!pathsByBlockerCount.ContainsKey(1))
                        pathsByBlockerCount[1] = new List<Vector2Int>();
                        
                    pathsByBlockerCount[1].Add(new Vector2Int(gX, gY));
                    LogAction($"Found single blocker solution: [{gX},{gY}]");
                }
            }
        }
        
        // If we found any single-blocker solutions, pick the one closest to the passenger
        if (pathsByBlockerCount.ContainsKey(1))
        {
            // Prioritize adjacent blockers first (they're more likely to be direct obstacles)
            var adjacentBlockers = pathsByBlockerCount[1]
                .Where(b => (Math.Abs(b.x - x) + Math.Abs(b.y - y)) == 1)
                .ToList();
                
            if (adjacentBlockers.Count > 0)
            {
                LogAction("Found adjacent single blocker solution - optimal!");
                return adjacentBlockers
                    .OrderBy(p => p.y)  // Sort by row (top to bottom)
                    .ThenBy(p => p.x)   // Then by column (left to right)
                    .Select(p => $"[x={p.x},y={p.y}]")
                    .ToList();
            }
            
            // Otherwise, return any single-blocker solution
            LogAction("Found single blocker solution - optimal!");
            return pathsByBlockerCount[1]
                .OrderBy(p => p.y)  // Sort by row (top to bottom)
                .ThenBy(p => p.x)   // Then by column (left to right)
                .Select(p => $"[x={p.x},y={p.y}]")
                .ToList();
        }
        
        // APPROACH 2: Check for two-blocker solutions
        // We'll try pairs of blockers
        for (int b1Y = 0; b1Y < gridHeight; b1Y++)
        {
            for (int b1X = 0; b1X < gridWidth; b1X++)
            {
                // Skip non-passenger tiles
                if (grid[b1X, b1Y] == null || grid[b1X, b1Y].IsDisabled || grid[b1X, b1Y].CurrentPassenger == null)
                    continue;
                    
                for (int b2Y = 0; b2Y < gridHeight; b2Y++)
                {
                    for (int b2X = 0; b2X < gridWidth; b2X++)
                    {
                        // Skip same blocker or non-passenger tiles
                        if ((b1X == b2X && b1Y == b2Y) || 
                            grid[b2X, b2Y] == null || 
                            grid[b2X, b2Y].IsDisabled || 
                            grid[b2X, b2Y].CurrentPassenger == null)
                        {
                            continue;
                        }
                        
                        // Try removing both passengers
                        bool[,] tempGrid = (bool[,])occupied.Clone();
                        tempGrid[b1X, b1Y] = false;
                        tempGrid[b2X, b2Y] = false;
                        
                        // Check if this creates a path
                        if (HasAnyPathToTop(x, y, tempGrid))
                        {
                            // We found a two-blocker solution
                            if (!pathsByBlockerCount.ContainsKey(2))
                                pathsByBlockerCount[2] = new List<Vector2Int>();
                                
                            // Add both blockers, with the one closer to the passenger first
                            if (Math.Abs(b1X - x) + Math.Abs(b1Y - y) <= Math.Abs(b2X - x) + Math.Abs(b2Y - y))
                            {
                                pathsByBlockerCount[2].Add(new Vector2Int(b1X, b1Y));
                                pathsByBlockerCount[2].Add(new Vector2Int(b2X, b2Y));
                            }
                            else
                            {
                                pathsByBlockerCount[2].Add(new Vector2Int(b2X, b2Y));
                                pathsByBlockerCount[2].Add(new Vector2Int(b1X, b1Y));
                            }
                            
                            LogAction($"Found two-blocker solution: [{b1X},{b1Y}] and [{b2X},{b2Y}]");
                            
                            // Once we find a solution, we can stop searching for this pair
                            break;
                        }
                    }
                }
            }
        }
        
        // If we found any two-blocker solutions, pick the first one
        if (pathsByBlockerCount.ContainsKey(2))
        {
            LogAction("Using two-blocker solution");
            return pathsByBlockerCount[2]
                .OrderBy(p => p.y)  // Sort by row (top to bottom)
                .ThenBy(p => p.x)   // Then by column (left to right)
                .Select(p => $"[x={p.x},y={p.y}]")
                .Take(2)  // Just take the first solution we found
                .ToList();
        }
        
        // APPROACH 3: Fallback to the old method to check all possible blockers
        LogAction("Fallback to checking all possible blockers");
        List<Vector2Int> allBlockers = new List<Vector2Int>();
        
        // Check direct vertical path
        List<Vector2Int> verticalBlockers = new List<Vector2Int>();
        for (int testY = y - 1; testY >= 0; testY--)
        {
            if (occupied[x, testY] && grid[x, testY]?.CurrentPassenger != null)
            {
                verticalBlockers.Add(new Vector2Int(x, testY));
                LogAction($"Vertical path blocker found at [{x},{testY}]");
            }
        }
        
        // Add all the blockers we've found
        allBlockers.AddRange(verticalBlockers);
        
        // Sort blockers by position (top to bottom, left to right)
        return allBlockers
            .OrderBy(p => p.y)  // Sort by row (top to bottom)
            .ThenBy(p => p.x)   // Then by column (left to right)
            .Select(p => $"[x={p.x},y={p.y}]")
            .ToList();
    }

    private void LogColorsInGameState()
    {
        StringBuilder sb = new StringBuilder();
        
        sb.AppendLine("===== COLOR ANALYSIS =====");
        
        // Log the current bus color
        if (currentBus != null)
        {
            Color busColor = currentBus.Color;
            string colorName = ColorToString(busColor); // Replace IdentifyColor with ColorToString
            sb.AppendLine($"Current bus color: {colorName} (R={busColor.r:F2}, G={busColor.g:F2}, B={busColor.b:F2})");
        }
        
        // Log grid passengers
        if (grid != null)
        {
            sb.AppendLine("\nGrid passengers:");
            int gridSizeX = grid.GetLength(0);
            int gridSizeY = grid.GetLength(1);
            
            for (int y = 0; y < gridSizeY; y++)
            {
                for (int x = 0; x < gridSizeX; x++)
                {
                    Tile tile = grid[x, y];
                    if (tile != null && !tile.IsDisabled && tile.CurrentPassenger != null)
                    {
                        Color passengerColor = tile.CurrentPassenger.Color;
                        string colorName = ColorToString(passengerColor); // Replace IdentifyColor with ColorToString
                        sb.AppendLine($"  Position [{x},{y}]: {colorName} (R={passengerColor.r:F2}, G={passengerColor.g:F2}, B={passengerColor.b:F2})");
                    }
                }
            }
        }
        
        // Log waiting area passengers
        if (waitingSlots != null)
        {
            sb.AppendLine("\nWaiting area passengers:");
            for (int i = 0; i < waitingSlots.Length; i++)
            {
                if (waitingSlots[i].Passenger != null)
                {
                    Color passengerColor = waitingSlots[i].Passenger.Color;
                    string colorName = ColorToString(passengerColor); // Replace IdentifyColor with ColorToString
                    sb.AppendLine($"  Slot {i}: {colorName} (R={passengerColor.r:F2}, G={passengerColor.g:F2}, B={passengerColor.b:F2})");
                }
                else
                {
                    sb.AppendLine($"  Slot {i}: Empty");
                }
            }
        }
        
        LogAction(sb.ToString());
    }
}

