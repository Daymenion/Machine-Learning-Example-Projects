using UnityEngine;

public class Passenger : MonoBehaviour
{
    public Color Color { get; private set; }
    private Material material;
    private GameManager manager;
    private bool nonPlayable;

    public void Initialize(Color color)
    {
        Color = color;
        material = GetComponent<Renderer>().material;
        material.color = color;
        manager = FindFirstObjectByType<GameManager>();
    }

    public void MakeNonPlayable()
    {
        this.nonPlayable = true;
    }

    private void OnMouseDown()
    {
        if (!nonPlayable)
            manager.OnPassengerClicked(this);
    }
}

