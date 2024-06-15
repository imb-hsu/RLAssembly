using System.Collections;
using System.Collections.Generic;
using System.Linq;

using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class DisassemblyAgent : Agent
{
    [SerializeField] private List<GameObject> parts;
    private List<bool> alreadyMoved; // Tracks if a part has been moved
    private List<bool> rewardRecieved; // Tracks if a part has been moved
    private int lastMovedPartId = -1; // -1 indicates no part has been moved yet

    private List<Vector3> initialPositions;
    private List<Vector3> initialLocalPositions;

    public void Start()
    {

    }

    public override void Initialize()
    {
        alreadyMoved = new List<bool>(new bool[parts.Count]);
        rewardRecieved = new List<bool>(new bool[parts.Count]);
        initialPositions = new List<Vector3>();
        initialLocalPositions = new List<Vector3>();

        Debug.unityLogger.logEnabled = false;

        Debug.Log("Init.");
        for (int i = 0; i < parts.Count; i++)
        {
            //Debug.Log("Part ID");
            //Debug.Log(i);
            //Debug.Log("Global Pos");
            //Debug.Log(parts[i].transform.position);
            initialPositions.Add(parts[i].transform.position);
            //Debug.Log("Local Pos");
            //Debug.Log(parts[i].transform.localPosition);
            initialLocalPositions.Add(parts[i].transform.localPosition);
        }
    }

    public override void OnEpisodeBegin()
    {
        Debug.Log("New Episode Start");
        lastMovedPartId = -1; // Reset at the beginning of each episode
        for (int i = 0; i < parts.Count; i++)
        {
            //Debug.Log("Part ID");
            //Debug.Log(i);
            alreadyMoved[i] = false; // Reset which parts were moved
            rewardRecieved[i] = false; // and which recieved rewards

            Rigidbody partRb = parts[i].GetComponentInChildren<Rigidbody>();
            partRb.velocity = Vector3.zero; // Reset velocity
            partRb.angularVelocity = Vector3.zero; // Reset angular velocity
            //Debug.Log("Distance");
            //float dist = Vector3.Distance(parts[i].transform.position, initialPositions[i]);
            //Debug.Log(dist);
            //Debug.Log("Global Pos");
            parts[i].transform.position = initialPositions[i]; // Reset to initial position
            //Debug.Log(parts[i].transform.position);
            //Debug.Log("Local Pos");
            parts[i].transform.localPosition = initialLocalPositions[i]; // Reset to initial position
            //Debug.Log(parts[i].transform.localPosition);
            partRb.constraints = RigidbodyConstraints.FreezeAll;

        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int objectToMoveId = actions.DiscreteActions[0];

        // Check if the part has already been moved and is not the same as the last moved part
        /*if (alreadyMoved[objectToMoveId] && lastMovedPartId != objectToMoveId)
        {
            AddReward(-0.0001f);
            return; // Skip if this part has already been moved and is not the last moved part
        }*/

        AddReward(-0.0001f);
        int moveDirection = actions.DiscreteActions[1];
        var moveSpeed = 80f; 
        Vector3 movementVector = Vector3.zero;
        switch (moveDirection)
        {
            case 0: movementVector = new Vector3(moveSpeed, 0, 0); break;
            case 1: movementVector = new Vector3(0, moveSpeed, 0); break;
            case 2: movementVector = new Vector3(0, 0, moveSpeed); break;
        }

        // Move part
        lastMovedPartId = objectToMoveId; // Update last moved part
        alreadyMoved[objectToMoveId] = true; // Mark this part as moved

        foreach (GameObject part in parts)
        {
            part.GetComponentInChildren<Rigidbody>().constraints = RigidbodyConstraints.FreezeAll;
        }
        Rigidbody rb = parts[objectToMoveId].GetComponentInChildren<Rigidbody>();
        rb.constraints = RigidbodyConstraints.None;
        rb.constraints = RigidbodyConstraints.FreezeRotation;
        parts[objectToMoveId].transform.position += movementVector * Time.deltaTime;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        for (int i = 0; i< parts.Count; i++)
        {
            sensor.AddObservation(parts[i].transform.position);
            sensor.AddObservation(rewardRecieved[i]);
            sensor.AddObservation(alreadyMoved[i]);
        }
        sensor.AddObservation(lastMovedPartId);
    }

    public void Update()
    {
        
        Vector3 center = new Vector3(0,0,0);
        int numPositions = parts.Count;
        for(int i = 0; i< numPositions; i++){
            center += parts[i].transform.position;
        }
        center /= numPositions;
        

        /*Debug.Log("All Current Rewards");
        foreach(var rewardRec in rewardRecieved)
        {
            Debug.Log(rewardRec);
        } */

        for (int i = 0; i < parts.Count; i++)
        {
            float dist = Vector3.Distance(parts[i].transform.position, initialPositions[i]);
            float distCenter = Vector3.Distance(parts[i].transform.position, center);

            if (dist > 2 && distCenter > 2 && !rewardRecieved[i]) {
            //if (dist > 2 && !rewardRecieved[i]) {
                AddReward(0.1f);
                rewardRecieved[i] = true;
                Debug.Log("Reward added for Part ID and Distance");
                Debug.Log(i);
                Debug.Log(dist);
            }
        }

        if ( !rewardRecieved.Contains(false) ) {
            
            Debug.Log("All Rewards Collected");
            EndEpisode();

        }
    }
}
