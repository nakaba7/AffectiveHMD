using System.Collections;
using System.Collections.Generic;
using UnityEngine;
 
public class AroundMove : MonoBehaviour
{
    [SerializeField] GameObject target;
    private float timer;
 
    float angle = 40;
    void Start(){
        timer = 0;
    }
 
    void Update () {
        timer += Time.deltaTime;
        Debug.Log(timer);
        if(timer <= 0.75f) transform.RotateAround (target.transform.position, Vector3.right, angle * Time.deltaTime);
        else if(timer <= 2.25f) transform.RotateAround (target.transform.position, Vector3.right, -angle * Time.deltaTime);
        else if(timer <= 3.0f) transform.RotateAround (target.transform.position, Vector3.right, angle * Time.deltaTime);
        else timer=0;       
    }
}