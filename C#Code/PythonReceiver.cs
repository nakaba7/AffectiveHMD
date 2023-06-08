using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.IO;

public class PythonReceiver : MonoBehaviour {

    static UdpClient udp;
    IPEndPoint remoteEP = null;
    // System.IO
    private StreamWriter sw;
    private SerialHandler serialHandler;
    private Instruction instruction;

    
    // Use this for initialization
    void Start () {
        int LOCA_LPORT = 50007;
        udp = new UdpClient(LOCA_LPORT);
        udp.Client.ReceiveTimeout = 2000;
        sw = new StreamWriter(@"C:\Users\yukin\Downloads\DemoSensorReceiver.csv", false, Encoding.GetEncoding("Shift_JIS"));
        sw.WriteLine("0");
    }

    // Update is called once per frame
    void Update ()
    {
        if(Input.GetKeyDown(KeyCode.Q)){
            FileClose();
            Debug.Log("File Closed");
        }
        else if(Input.GetKey(KeyCode.Return)){
            sw.WriteLine("1");
        }
        else{
            sw.WriteLine("0");
        }
        IPEndPoint remoteEP = null;
        
        try{
            byte[] data = udp.Receive(ref remoteEP);
            string text = Encoding.UTF8.GetString(data);
            Debug.Log(text);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning(e.Message);
        }
        
    }

     public void FileClose()
    {
        sw.Close();
    }

    
}