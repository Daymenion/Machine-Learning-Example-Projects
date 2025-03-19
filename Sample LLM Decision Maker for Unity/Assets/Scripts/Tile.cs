using UnityEngine;


public class Tile : MonoBehaviour
{
    public Transform PassengerAnchor;

    public GameObject TileVisual;
    public Passenger CurrentPassenger { get; private set; }
    private bool isDisabled;
    public bool IsDisabled { get{ return isDisabled;} }

    public void SetPassenger(Passenger passenger)
    {
        CurrentPassenger = passenger;
        passenger.transform.SetParent(PassengerAnchor);
    }

    public void ClearPassenger()
    {
        CurrentPassenger = null;
    }

    public void SetDisabled(bool isDisabled)
    {
        TileVisual.SetActive(!isDisabled);
        this.isDisabled = isDisabled;
    }
}
