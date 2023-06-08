using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Random = UnityEngine.Random;

public class SeriesInstruction : MonoBehaviour
{
    private SerialHandler serialHandler;
    private SeriesCsvWriter SeriesCsvWriter;
    [SerializeField] private GameObject emotionTextObject = null; // Text�I�u�W�F�N�g
    [SerializeField] private GameObject countDownTimerObject = null;//�v���c�莞�Ԃ�\������
    private Text emotionText;//�v�����̕\��\��
    private Text timerText;//�c�莞�ԕ\��
    private float countDownTimer;//�J�E���g�_�E���^�C�}�[
    private bool startFlag;
    [SerializeField] private float captureTime;//��������5�b�����킹���f�[�^���W����
    private float timer;//�o�ߎ���
    private float currentTime;//�v���Ԋu���v��p�̃^�C�}�[
    [SerializeField] private float span;//�Z���T�l�̌v���Ԋu
    private int emotionPattern;//�����ɂ���Čv������\����Ǘ�����
    private string[] emotionList = { "Neutral", "Smile", "Surprised", "Sad", "Angry" };
    [SerializeField] private GameObject instructionTextObject;
    private Text instructionText;//���̕\��\��
    private HMDDataCapture hmdDataCapture;
    private bool isFirstCapture;
    private bool randChangeFlag;//�\������߂闐�����X�V���邩�ǂ���
    private int tmpRandSaver;
    private int prev_emotionPattern;
    private int captureNum;
    private int counter;
    [SerializeField] private GameObject countTextObject;
    private Text countText;
    [SerializeField] private float announceTime;
    private int[] emotionOrder = { 0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 4, 0 };//�\��J�ڂ̏���
    private float firstCaptureTime;
    private float originalCaptureTime;
    [SerializeField] private GameObject currentNeutral;
    [SerializeField] private GameObject currentSmile;
    [SerializeField] private GameObject currentSurprised;
    [SerializeField] private GameObject currentSad;
    [SerializeField] private GameObject currentAngry;

    [SerializeField] private GameObject nextNeutral;
    [SerializeField] private GameObject nextSmile;
    [SerializeField] private GameObject nextSurprised;
    [SerializeField] private GameObject nextSad;
    [SerializeField] private GameObject nextAngry;



    /*
        Neutral : 0
        Smile : 1
        Surprised : 2
        Sad : 3
        Angry : 4
     */

    // Start is called before the first frame update
    void Start()
    {
        AllMainEmotionDisabled();
        AllNextEmotionDisabled();
        serialHandler = this.GetComponent<SerialHandler>();
        SeriesCsvWriter = this.GetComponent<SeriesCsvWriter>();
        hmdDataCapture = this.GetComponent<HMDDataCapture>();
        //SetTimer();
        startFlag = false;
        emotionText = emotionTextObject.GetComponent<Text>();
        timerText = countDownTimerObject.GetComponent<Text>();
        instructionText = instructionTextObject.GetComponent<Text>();
        emotionPattern = 0;
        prev_emotionPattern = 0;
        timer = 0;
        //instructionText.text = $"Next emotion is {emotionList[0]}";
        isFirstCapture = true;
        randChangeFlag = true;
        counter = 0;
        countText = countTextObject.GetComponent<Text>();
        captureNum = emotionOrder.Length;
        countText.text = $"{counter}/{captureNum}";
        originalCaptureTime = captureTime;
        firstCaptureTime = originalCaptureTime + 2.0f;
        captureTime = firstCaptureTime;
        SetTimer();
        currentNeutral.SetActive(true);
        nextNeutral.SetActive(true);
    }

    private void SetTimer()//�J�E���g�_�E���^�C�}�[�̏�����
    {
        countDownTimer = captureTime;
    }

    public bool getIsFirstCapture()
    {
        return isFirstCapture;
    }

    private void AllMainEmotionDisabled()
    {
        currentNeutral.SetActive(false);
        currentSmile.SetActive(false);
        currentSurprised.SetActive(false);
        currentSad.SetActive(false);
        currentAngry.SetActive(false);
    }

    private void AllNextEmotionDisabled()
    {
        nextNeutral.SetActive(false);
        nextSmile.SetActive(false);  
        nextSurprised.SetActive(false);
        nextSad.SetActive(false);
        nextAngry.SetActive(false);
    }
    private void ChangeCurrentEmotionImage(int emotionPattern)
    {
        AllMainEmotionDisabled();
        switch (emotionPattern)
        {
            case 0:
                currentNeutral.SetActive(true);
                break;
            case 1:
                currentSmile.SetActive(true);
                break;
            case 2:
                currentSurprised.SetActive(true);
                break;
            case 3:
                currentSad.SetActive(true);
                break;
            case 4:
                currentAngry.SetActive(true);
                break;
            default:
               
                break;
        }
    }
    private void ChangeNextEmotionImage(int emotionPattern)
    {
        AllNextEmotionDisabled();
        switch (emotionPattern)
        {
            case 0:
                nextNeutral.SetActive(true);
                break;
            case 1:
                nextSmile.SetActive(true);
                break;
            case 2:
                nextSurprised.SetActive(true);
                break;
            case 3:
                nextSad.SetActive(true);
                break;
            case 4:
                nextAngry.SetActive(true);
                break;
            default:

                break;
        }
    }
    // Update is called once per frame
    void Update()
    {
        //Debug.Log(countDownTimer);
        if (Input.GetKeyDown(KeyCode.Return))//EnterKey����������v���J�n
        {
            //instructionTextObject.SetActive(false);
            startFlag = true;
            instructionTextObject.SetActive(false);
        }

        //isFirstCapture = true;
        if (startFlag)
        {
            timer += Time.deltaTime;//���݂̕\����v�����鎞��
            currentTime += Time.deltaTime;//�Z���T�l���擾���Ă��玟�ɃZ���T�l���擾����܂ł̃^�C�}�[
            captureSensor(emotionPattern);
            //captureSensor("Smile");
        }


    }
    

    public void captureSensor(int emotionNum)//emotion���w�肵�ăZ���T�l���擾���Ccsv�t�@�C���֏�������
    {
        //emotionText.text = emotion;//��ʂɕ\������\��ύX
        ChangeCurrentEmotionImage(emotionNum);
        if (currentTime >= span)//�Z���T�l��1�Z�b�g�擾��ɓ���
        {
            /*serialHandler.UpdateSensorData();//�Z���T�l�擾
            SeriesCsvWriter.SaveSensorData(serialHandler.getSensorData(), emotionList[emotionNum], hmdDataCapture.getHMDPos(), hmdDataCapture.getHMDRotation());//csv�t�@�C���֕ۑ�*/

            serialHandler.UpdateStringSensorData();
            SeriesCsvWriter.SaveStringSensorData(serialHandler.getStringSensorData(), emotionList[emotionNum], hmdDataCapture.getHMDPos(), hmdDataCapture.getHMDRotation());

            currentTime = 0;
        }
        if (timer < captureTime)//�v�����Ԉȉ��̏ꍇ��span�b���ƂɃZ���T�l���擾����csv�t�@�C���ɕۑ�
        {
            countDownTimer -= Time.deltaTime;
            timerText.text = countDownTimer.ToString("f1");//��ʂɎc�莞�ԕ\��
            if (timer > captureTime - originalCaptureTime) isFirstCapture = false;
            if(timer + announceTime >= captureTime)//�\��؂�ւ�2�b�O�ɉ�ʂɎ��̕\���\��
            {
                /*if (randChangeFlag)
                {
                    tmpRandSaver = Random.Range(0, 5);
                    while (tmpRandSaver == prev_emotionPattern)//1�O�Ɠ����\��ł���΂ق��̕\��ɕς���
                    {
                        tmpRandSaver = Random.Range(0, 5);
                    }
                    randChangeFlag = false;
                }*/
                if (counter < emotionOrder.Length - 1) ChangeNextEmotionImage(emotionOrder[counter + 1]);
                else 
                { 
                    instructionText.text = "This is the last emotion.";
                    AllNextEmotionDisabled();
                }
                instructionTextObject.SetActive(true);
            }

            /*if (currentTime >= span)//�Z���T�l��1�Z�b�g�擾��ɓ���
            {
                serialHandler.UpdateSensorData();//�Z���T�l�擾
                SeriesCsvWriter.SaveSensorData(serialHandler.getSensorData(), emotion, hmdDataCapture.getHMDPos(), hmdDataCapture.getHMDRotation());//csv�t�@�C���֕ۑ�
                currentTime = 0;
            }*/
        }
        else
        {
            //startFlag = false;
            captureTime = originalCaptureTime;
            SetTimer();
            //randChangeFlag = true;
            timer = 0;
            currentTime = 0;
            counter++;
            instructionTextObject.SetActive(false);
            if(counter < emotionOrder.Length) emotionPattern = emotionOrder[counter];
            //prev_emotionPattern = emotionPattern;
            isFirstCapture = false;
           
            countText.text = $"{counter}/{captureNum}";
            if (counter == captureNum)
            {
                startFlag = false;
                instructionText.text = "Finish!";
                instructionTextObject.SetActive(true);
                SeriesCsvWriter.WriteEndToken();
                SeriesCsvWriter.FileClose();
                serialHandler.Close();
            }

        }
    }
}