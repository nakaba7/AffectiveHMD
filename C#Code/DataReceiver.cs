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
        //�M������M�����Ƃ��ɁA���̃��b�Z�[�W�̏������s��
        serialHandler.OnDataReceived += OnDataReceived;
    }

    void OnDataReceived(string message)
    {
        /*var data = message.Split(
                new string[] { "\n" }, System.StringSplitOptions.None);
        try
        {
            Debug.Log(data[0]);//Unity�̃R���\�[���Ɏ�M�f�[�^��\��

            SensorList.Add(int.Parse(data[0]));//string�œ��͂��󂯎��̂�int�փL���X�g����i�Z���T�l��float�Ȃ�float�֕ς���j
        }
        catch (System.Exception e)
        {
            Debug.LogWarning(e.Message);//�G���[��\��
        }*/
        Debug.Log(message);
    }

    public List<int> getSensorData()//�����ς݂̃Z���T�l���X�g��Ԃ�
    {
        return values;
    }

    public void sensorValuesChange()//�Z���T�l�ɑ΂���r�b�g���Z
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
