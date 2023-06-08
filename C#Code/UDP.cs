using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class UDP : MonoBehaviour {
    [SerializeField] private GameObject LSTMNeutral;
    [SerializeField] private GameObject LSTMSmile;
    [SerializeField] private GameObject LSTMSurprised;
    [SerializeField] private GameObject LSTMSad;
    [SerializeField] private GameObject LSTMAngry;

    [SerializeField] private GameObject DNNNeutral;
    [SerializeField] private GameObject DNNSmile;
    [SerializeField] private GameObject DNNSurprised;
    [SerializeField] private GameObject DNNSad;
    [SerializeField] private GameObject DNNAngry;


    static UdpClient udp;
    IPEndPoint remoteEP = null;
    int i = 0;
    // Use this for initialization

    private void AllMainEmotionDisabled()
    {
        LSTMNeutral.SetActive(false);
        LSTMSmile.SetActive(false);
        LSTMSurprised.SetActive(false);
        LSTMSad.SetActive(false);
        LSTMAngry.SetActive(false);

        DNNNeutral.SetActive(false);
        DNNSmile.SetActive(false);
        DNNSurprised.SetActive(false);
        DNNSad.SetActive(false);
        DNNAngry.SetActive(false);
    }

    void Start () {
        int LOCA_LPORT = 50007;

        udp = new UdpClient(LOCA_LPORT);
        udp.Client.ReceiveTimeout = 2000;
    }

    // Update is called once per frame
    void Update ()
    {
        IPEndPoint remoteEP = null;
        byte[] data = udp.Receive(ref remoteEP);
        string text = Encoding.UTF8.GetString(data);
        
        AllMainEmotionDisabled();
        switch (text[0])
        {
            case '0':
                LSTMNeutral.SetActive(true);
                break;
            case '1':
                LSTMSmile.SetActive(true);
                break;
            case '2':
                LSTMSurprised.SetActive(true);
                break;
            case '3':
                LSTMSad.SetActive(true);
                break;
            case '4':
                LSTMAngry.SetActive(true);
                break;
            default:
                break;
        }

        switch (text[1])
        {
            case '0':
                DNNNeutral.SetActive(true);
                break;
            case '1':
                DNNSmile.SetActive(true);
                break;
            case '2':
                DNNSurprised.SetActive(true);
                break;
            case '3':
                DNNSad.SetActive(true);
                break;
            case '4':
                DNNAngry.SetActive(true);
                break;
            default:
                break;
        }

        Debug.Log(text);
        Debug.Log("LSTM"+text[0]);
        Debug.Log("DNN"+text[1]);
    }
}