using UnityEngine;
using System.Collections;
using System.IO.Ports;
using System.Collections.Generic;
public class DataReceiver : MonoBehaviour
{
    public SerialHandler serialHandler;

    /*private SerialPort serialPort;
    [SerializeField] private string portName;
    [SerializeField] private int baudRate;
    [SerializeField] private int timeOut = 1;*/
    private List<int> values = new List<int>() { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    private List<int> SensorList = new List<int>(24);
    public char startChar = 'A';
    public char stopChar = 'Y';


    void Start()
    {
        //信号を受信したときに、そのメッセージの処理を行う
        serialHandler.OnDataReceived += OnDataReceived;
    }

    void OnDataReceived(string message)
    {
        /*var data = message.Split(
                new string[] { "\n" }, System.StringSplitOptions.None);
        try
        {
            Debug.Log(data[0]);//Unityのコンソールに受信データを表示

            SensorList.Add(int.Parse(data[0]));//stringで入力を受け取るのでintへキャストする（センサ値がfloatならfloatへ変える）
        }
        catch (System.Exception e)
        {
            Debug.LogWarning(e.Message);//エラーを表示
        }*/
        Debug.Log(message);
    }

    public List<int> getSensorData()//処理済みのセンサ値リストを返す
    {
        return values;
    }

    public void sensorValuesChange()//センサ値に対するビット演算
    {
        if ((SensorList.Count == 24) && (SensorList[0] == startChar) && (SensorList[23] == stopChar))
        {
            values[0] = (((SensorList[1]) & 0xff) << 2) + (((SensorList[9]) & 0xc0) >> 6);
            values[1] = (((SensorList[2]) & 0xff) << 2) + (((SensorList[9]) & 0x30) >> 4);
            values[2] = (((SensorList[3]) & 0xff) << 2) + (((SensorList[9]) & 0x0c) >> 2);
            values[3] = (((SensorList[4]) & 0xff) << 2) + ((SensorList[9]) & 0x03);
            values[4] = (((SensorList[5]) & 0xff) << 2) + (((SensorList[10]) & 0xc0) >> 6);
            values[5] = (((SensorList[6]) & 0xff) << 2) + (((SensorList[10]) & 0x30) >> 4);
            values[6] = (((SensorList[7]) & 0xff) << 2) + (((SensorList[10]) & 0x0c) >> 2);
            values[7] = (((SensorList[8]) & 0xff) << 2) + ((SensorList[10]) & 0x03);
            values[8] = (((SensorList[13]) & 0xff) << 2) + (((SensorList[21]) & 0xc0) >> 6);
            values[9] = (((SensorList[14]) & 0xff) << 2) + (((SensorList[21]) & 0x30) >> 4);
            values[10] = (((SensorList[15]) & 0xff) << 2) + (((SensorList[21]) & 0x0c) >> 2);
            values[11] = (((SensorList[16]) & 0xff) << 2) + ((SensorList[21]) & 0x03);
            values[12] = (((SensorList[17]) & 0xff) << 2) + (((SensorList[22]) & 0xc0) >> 6);
            values[13] = (((SensorList[18]) & 0xff) << 2) + (((SensorList[22]) & 0x30) >> 4);
            values[14] = (((SensorList[19]) & 0xff) << 2) + (((SensorList[22]) & 0x0c) >> 2);
            values[15] = (((SensorList[20]) & 0xff) << 2) + ((SensorList[22]) & 0x03);

            Debug.Log(values);
        }
        else
        {
            Debug.Log("sensor data error");
        }
    }   
    
}
