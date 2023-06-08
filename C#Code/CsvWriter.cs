using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Text;
using UnityEngine;

public class CsvWriter : MonoBehaviour
{
    // System.IO
    private StreamWriter sw;
    private SerialHandler serialHandler;
    private Instruction instruction;

    // Start is called before the first frame update
    void Start()
    {
        sw = new StreamWriter(@"SensorData.csv", false, Encoding.GetEncoding("Shift_JIS"));
        serialHandler = this.GetComponent<SerialHandler>();
        instruction = this.GetComponent<Instruction>();
    }

    // Update is called once per frame
    /*void Update()
    {
        // Enterキーが押されたらcsvへの書き込みを終了する
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

    public void SaveSensorData(List<int> values, string emotion, Vector3 hMDpos, Vector3 hMDRotation){
        List<string> tmplist = new List<string>();
        tmplist.Add(emotion);
        foreach(int data in values){
            
            tmplist.Add(data.ToString());
            //Debug.Log(data+" : "+data.ToString());
        }
        tmplist.Add(hMDpos.x.ToString());
        tmplist.Add(hMDpos.y.ToString());
        tmplist.Add(hMDpos.z.ToString());
        tmplist.Add(hMDRotation.x.ToString());
        tmplist.Add(hMDRotation.y.ToString());
        tmplist.Add(hMDRotation.z.ToString());
        string writelist = string.Join(",", tmplist);
        if(!(instruction.getIsFirstCapture())) sw.WriteLine(writelist); //最初のセンサ値でなければcsvファイルへ書き込み
    }
}
