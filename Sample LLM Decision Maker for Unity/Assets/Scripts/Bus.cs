using System.Collections.Generic;
using UnityEngine;

public class Bus : MonoBehaviour
{
    public Color Color { get; private set; }
    public bool IsFull => passengers.Count >= 3;
    
    private List<Passenger> passengers = new List<Passenger>();
    [SerializeField] private GameObject busModel;
    private Material material;
    [SerializeField]
    private Transform[] seatPositions;
    

    public void Initialize(Color color)
    {
        Color = color;
        material = busModel.GetComponent<Renderer>().material;
        material.color = color;
    }

    public void AddPassenger(Passenger passenger)
    {
        if (IsFull) return;

        int seatIndex = passengers.Count;
        
        StartCoroutine(MovePassengerToBus(passenger, seatPositions[seatIndex].position));
        
        passengers.Add(passenger);
    }
    
    
    private System.Collections.IEnumerator MovePassengerToBus(Passenger passenger, Vector3 targetPos)
    {
        float moveTime = 0.2f;
        float elapsed = 0;
        Vector3 startPos = passenger.transform.position;

        while (elapsed < moveTime)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / moveTime;
            passenger.transform.position = Vector3.Lerp(startPos, targetPos, t);
            yield return null;
        }
        
        passenger.transform.SetParent(transform);
        passenger.transform.position = targetPos;
        
       
        
    }
}
