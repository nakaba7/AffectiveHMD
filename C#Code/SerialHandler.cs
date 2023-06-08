using UnityEngine;
using System.Collections;
using System.IO.Ports;
using System.Threading;
using System.Collections.Generic;
using UnityEngine.UI;
public class SerialHandler : MonoBehaviour
{
    public delegate void SerialDataReceivedEventHandler(string message);
    public event SerialDataReceivedEventHandler OnDataReceived;
    public string portName = "COM4";
    public int baudRate = 115200;
    public GameObject text = null; // Textオブジェクト
    private SerialPort serialPort_;
    private bool isRunning_ = false;
    private string message_;
    private bool isNewMessageReceived_ = false;
    private byte[] inputBytes = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    private List<int> values = new List<int>() { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    private List<int> SensorList;
    private char startChar = 'A';
    private char stopChar = 'Y';
    private CsvWriter csvWriter;


    void Start()
    {
        csvWriter = this.GetComponent<CsvWriter>();
    }
    void Awake()
    {
        Open();
    }

    /*void FixedUpdate()
    {
        UpdateSensorData();
    }*/

    void OnDestroy()
    {
        Close();
    }

    private void Open()
    {
        serialPort_ = new SerialPort(portName, baudRate, Parity.None, 8, StopBits.One);
        //�܂���
        //serialPort_ = new SerialPort(portName, baudRate);
        serialPort_.Open();

        isRunning_ = true;
        serialPort_.DiscardInBuffer();//���̓o�b�t�@�̃��Z�b�g
        Debug.Log("Port Open!");
    }

    public void Close()
    {
        isNewMessageReceived_ = false;
        isRunning_ = false;

        if (serialPort_ != null && serialPort_.IsOpen)
        {
            serialPort_.Close();
            serialPort_.Dispose();
        }
    }

    private void Read()
    {
        if (isRunning_ && serialPort_ != null && serialPort_.IsOpen)
        {
            try
            {
                //message_ = serialPort_.ReadLine();
                serialPort_.Read(inputBytes, 0, 24);
                isNewMessageReceived_ = true;

            }
            catch (System.Exception e)
            {
                Debug.LogWarning(e.Message);
            }
        }
        else Debug.Log("Not Read");
    }

    public void Write(string message)
    {
        try
        {
            byte[] data = System.Text.Encoding.GetEncoding("shift_jis").GetBytes(message);//"b"���o�C�g��֕ϊ�
            serialPort_.Write(data, 0, data.Length);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning(e.Message);
        }
    }

    private void setSensorList()
    {
        SensorList = new List<int>();
        for (int i = 0; i < inputBytes.Length; i++)
        {
            SensorList.Add(inputBytes[i]);
        }

    }

    public void sensorValuesChange()//�Z���T�l�ɑ΂���r�b�g���Z
    {
        if ((SensorList.Count == 24) && (SensorList[0] == startChar) && (SensorList[23] == stopChar))
        {
            //Debug.Log("Change");
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
        }
        else
        {
            Debug.Log("sensor data error");
        }
    }

    public void UpdateSensorData()//�Z���T�l�̍X�V
    {
        Write("b");//�ʐM��̃R���s���[�^���Z���T�l�𑗂鍇�}�ƂȂ镶���𑗂�ib�Ƃ���������������ƃZ���T�l���Ԃ��Ă���j
        Read();//�Z���T�l�ǂݍ���
        setSensorList();//���̓Z���T�l�����X�g�֊i�[
        sensorValuesChange();//�r�b�g���Z����
        //csvWriter.SaveSensorData(values, "Smile");
        Text sensorValueText = text.GetComponent<Text>();
        sensorValueText.text = values[0] + ", " + values[1] + ", " + values[2] + ", " + values[3] + ", " + values[4] + ", " + values[5] + ", " + values[6] + ", " + values[7] + "\n" + values[8] + ", " + values[9] + ", " + values[10] + ", " + values[11] + ", " + values[12] + ", " + values[13] + ", " + values[14] + ", " + values[15];
        serialPort_.DiscardInBuffer();//���̓o�b�t�@�̃��Z�b�g
        serialPort_.DiscardOutBuffer();//�o�̓o�b�t�@�̃��Z�b�g
    }

    private string stringValues = "0";

    public void UpdateStringSensorData()
    {
        Write("a");
        if (isRunning_ && serialPort_ != null && serialPort_.IsOpen)
        {
            try
            {
                
                stringValues = serialPort_.ReadLine();
                isNewMessageReceived_ = true;

            }
            catch (System.Exception e)
            {
                Debug.LogWarning(e.Message);
            }
        }
        else Debug.Log("Not Read");

        stringValues = stringValues.Replace(":", ",");

        serialPort_.DiscardInBuffer();//���̓o�b�t�@�̃��Z�b�g
        serialPort_.DiscardOutBuffer();//�o�̓o�b�t�@�̃��Z�b�g       
    }

    public string getStringSensorData()
    {
        return stringValues;
    }

    public List<int> getSensorData()//�����ς݂̃Z���T�l���X�g��Ԃ�
    {
        return values;
    }

    public void printSensorData()
    {
        Debug.Log("values : [" + values[0] + ", " + values[1] + ", " + values[2] + ", " + values[3] + ", " + values[4] + ", " + values[5] + ", " + values[6] + ", " + values[7] + ", " + values[8] + ", " + values[9] + ", " + values[10] + ", " + values[11] + ", " + values[12] + ", " + values[13] + ", " + values[14] + ", " + values[15] + "]");
    }
}