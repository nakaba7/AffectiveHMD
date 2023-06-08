using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using System.Text;


public class SeriesCsvWriter : MonoBehaviour
{
    // System.IO
    private StreamWriter sw;
    private SerialHandler serialHandler;
    private SeriesInstruction SeriesInstruction;
    [SerializeField] private string userName;

    // Start is called before the first frame update
    void Start()
    {
        DateTime dt = DateTime.Now;
        String name = dt.ToString($"{dt:yyyyMMddHHmmss}");
        name = userName + "_" + name;
        sw = new StreamWriter($"CaptureData_csv\\{name}.csv", false, Encoding.GetEncoding("Shift_JIS"));
        serialHandler = this.GetComponent<SerialHandler>();
        SeriesInstruction = this.GetComponent<SeriesInstruction>();
    }

    // Update is called once per frame
    /*void Update()
    {
        // Enter�L�[�������ꂽ��csv�ւ̏������݂��I������
        if (Input.GetKeyDown(KeyCode.Return))
        {
            serialHandler.Close();
            sw.Close();
            Debug.Log("File Closed!");
        }
    }*/

    public void FileClose()
    {
        sw.Close();
    }

    public void SaveSensorData(List<int> values, string emotion, Vector3 hMDpos, Vector3 hMDRotation)
    {
        
        List<string> tmplist = new List<string>();
        //tmplist.Add(emotion);
        /*
        Neutral : 0
        Smile : 1
        Surprised : 2
        Sad : 3
        Angry : 4
         */
        switch (emotion)
        {
            case "Neutral":
                tmplist.Add("0");
                break;
            case "Smile":
                tmplist.Add("1");
                break;
            case "Surprised":
                tmplist.Add("2");
                break;
            case "Sad":
                tmplist.Add("3");
                break;
            case "Angry":
                tmplist.Add("4");
                break;

        }
        foreach (int data in values)
        {

            tmplist.Add(data.ToString());
            //Debug.Log(data+" : "+data.ToString());
        }
        /*tmplist.Add(hMDpos.x.ToString());
        tmplist.Add(hMDpos.y.ToString());
        tmplist.Add(hMDpos.z.ToString());*/
        tmplist.Add(hMDRotation.x.ToString());
        tmplist.Add(hMDRotation.y.ToString());
        tmplist.Add(hMDRotation.z.ToString());
        tmplist.Add(hMDpos.x.ToString());
        tmplist.Add(hMDpos.y.ToString());
        tmplist.Add(hMDpos.z.ToString());
        string writelist = string.Join(",", tmplist);
        if (!(SeriesInstruction.getIsFirstCapture())) sw.WriteLine(writelist); //�ŏ��̃Z���T�l�łȂ����csv�t�@�C���֏�������
    }
    public void WriteEndToken()
    {
        sw.WriteLine('a');
        Debug.Log("a written!");

    }

    public void SaveStringSensorData(string stringSensorData, string emotion, Vector3 hMDpos, Vector3 hMDRotation)
    {
        List<string> tmplist = new List<string>();
        //tmplist.Add(emotion);
        /*
        Neutral : 0
        Smile : 1
        Surprised : 2
        Sad : 3
        Angry : 4
         */
         
        switch (emotion)
        {
            case "Neutral":
                tmplist.Add("0");
                break;
            case "Smile":
                tmplist.Add("1");
                break;
            case "Surprised":
                tmplist.Add("2");
                break;
            case "Sad":
                tmplist.Add("3");
                break;
            case "Angry":
                tmplist.Add("4");
                break;

        }
        tmplist.Add(stringSensorData);
        /*foreach (int data in values)
        {

            tmplist.Add(data.ToString());
            //Debug.Log(data+" : "+data.ToString());
        }*/
       
        tmplist.Add(hMDRotation.x.ToString());
        tmplist.Add(hMDRotation.y.ToString());
        tmplist.Add(hMDRotation.z.ToString());
        tmplist.Add(hMDpos.x.ToString());
        tmplist.Add(hMDpos.y.ToString());
        tmplist.Add(hMDpos.z.ToString());
        string writelist = string.Join(",", tmplist);
        if (!(SeriesInstruction.getIsFirstCapture())) sw.WriteLine(writelist); 
    }
}
