[System.Serializable]
public class LevelData
{
    public int ID;
    public int gridX;
    public int gridY;
    public CellRow[] cellMatrix;
    public BusConfig[] busConfigs;
    public CameraPreset cameraPreset;
    public float levelTime;


    public int TotalPassengerCount()
    {
        int count = 0;
        foreach (var row in cellMatrix)
        {
            foreach (var tile in row.cellArray)
            {
                if (tile.passengers.Length > 0)
                    count++;
            }
        }

        return count;
    }
}

[System.Serializable]
public class CellRow
{
    public Cell[] cellArray;
}

[System.Serializable]
public class Cell
{
    public PassengerData[] passengers;
    public bool isDisabled;
}

[System.Serializable]
public class PassengerData
{
    public int color;
    
}

[System.Serializable]
public class BusConfig
{
    public int color;
    public int passengerCapacity;
}

[System.Serializable]
public class CameraPreset
{
    public Vector3Data cameraPosition;
    public Vector3Data cameraRotation;
    public float cameraFov;
}

[System.Serializable]
public class Vector3Data
{
    public float x;
    public float y;
    public float z;
}