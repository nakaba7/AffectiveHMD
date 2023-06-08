using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Instruction : MonoBehaviour
{
    private SerialHandler serialHandler;
    private CsvWriter csvWriter;
    [SerializeField] private GameObject emotionTextObject = null; // Textオブジェクト
    [SerializeField] private GameObject countDownTimerObject = null;//計測残り時間を表示する
    private Text emotionText;//計測中の表情表示
    private Text timerText;//残り時間表示
    private float countDownTimer;//カウントダウンタイマー
    private bool startFlag;
    [SerializeField] private float captureTime;//準備時間5秒を合わせたデータ収集時間
    private float timer;//経過時間
    private float currentTime;//計測間隔を計る用のタイマー
    [SerializeField] private float span;//センサ値の計測間隔
    private int emotionPattern;//数字によって計測する表情を管理する
    private string[] emotionList = { "Neutral", "Smile", "Sad", "Surprised", "Angry" };
    [SerializeField] private GameObject instructionTextObjcect;
    private Text instructionText;//次の表情表示
    private HMDDataCapture hmdDataCapture;
    private bool isFirstCapture;

    // Start is called before the first frame update
    void Start()
    {
        serialHandler = this.GetComponent<SerialHandler>();
        csvWriter = this.GetComponent<CsvWriter>();
        hmdDataCapture = this.GetComponent<HMDDataCapture>();
        SetTimer();
        startFlag = false;
        emotionText = emotionTextObject.GetComponent<Text>();
        timerText = countDownTimerObject.GetComponent<Text>();
        instructionText = instructionTextObjcect.GetComponent<Text>();
        emotionPattern = 0;
        timer = 0;
        instructionText.text = $"Next emotion is {emotionList[0]}";
        isFirstCapture = true;
    }

    private void SetTimer()//カウントダウンタイマーの初期化
    {
        countDownTimer = captureTime;
    }

    public bool getIsFirstCapture()
    {
        return isFirstCapture;
    }

    // Update is called once per frame
    void Update()
    {
        //Debug.Log(countDownTimer);
        if (Input.GetKeyDown(KeyCode.Return))//EnterKeyを押したら計測開始
        {
            instructionTextObjcect.SetActive(false);
            startFlag = true;
            isFirstCapture = true;
        }

        if (startFlag)
        {
            timer += Time.deltaTime;//秒数加算
            currentTime += Time.deltaTime;
            switch (emotionPattern)
            {
                case 0:
                    captureSensor(emotionList[0]);
                    break;
                case 1:
                    captureSensor(emotionList[1]);
                    break;
                case 2:
                    captureSensor(emotionList[2]);
                    break;
                case 3:
                    captureSensor(emotionList[3]);
                    break;
                case 4:
                    captureSensor(emotionList[4]);
                    break;
                default:
                    csvWriter.FileClose();
                    serialHandler.Close();
                    break;
            }
            //captureSensor("Smile");
        }
    }

    public void captureSensor(string emotion)//emotionを指定してセンサ値を取得し，csvファイルへ書きこみ
    {
        emotionText.text = emotion;//画面に表示する表情変更
        
        if (timer < captureTime)//計測時間以下の場合はspan秒ごとにセンサ値を取得してcsvファイルに保存
        {
            countDownTimer -= Time.deltaTime;
            timerText.text = countDownTimer.ToString("f1");//画面に残り時間表示
          
            if (currentTime >= span)
            {
                serialHandler.UpdateSensorData();//センサ値取得
                csvWriter.SaveSensorData(serialHandler.getSensorData(), emotion, hmdDataCapture.getHMDPos(), hmdDataCapture.getHMDRotation());//csvファイルへ保存
                isFirstCapture = false;
                currentTime = 0;
            }
        }
        else
        {
            startFlag = false;
            SetTimer();
            emotionPattern++;
            timer = 0;
            currentTime = 0;
            
            if (emotionPattern < emotionList.Length) instructionText.text = $"Next emotion is {emotionList[emotionPattern]}";
            else instructionText.text = "Finish!";
            instructionTextObjcect.SetActive(true);
        }
    }
}