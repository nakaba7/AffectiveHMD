using System;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.UI;

public class HMDDataCapture : MonoBehaviour
{
    //HMDの位置座標格納用
    private Vector3 HMDPosition;
    //HMDの回転座標格納用（クォータニオン）
    private Quaternion HMDRotationQ;
    //HMDの回転座標格納用（オイラー角）
    private Vector3 HMDRotation;
    [SerializeField] private GameObject hmdRotationTextObject;
    [SerializeField] private GameObject hmdPosTextObject;
    private Text hmdRotationText;
    private Text hmdPosText;

    //1フレーム毎に呼び出されるUpdateメゾット
    private void Start()
    {
        hmdRotationText = hmdRotationTextObject.GetComponent<Text>();
        hmdPosText = hmdPosTextObject.GetComponent<Text>();
    }
    void Update()
    {

        /*InputTracking.GetLocalPosition(XRNode.機器名)で機器の位置や向きを呼び出せる*/

        //Head（ヘッドマウンドディスプレイ）の情報を一時保管-----------
        //位置座標を取得
        HMDPosition = InputTracking.GetLocalPosition(XRNode.Head);
        //回転座標をクォータニオンで値を受け取る
        HMDRotationQ = InputTracking.GetLocalRotation(XRNode.Head);
        //取得した値をクォータニオン → オイラー角に変換
        HMDRotation = HMDRotationQ.eulerAngles;
        //--------------------------------------------------------------



        //取得したデータを表示（HMDP：HMD位置，HMDR：HMD回転，LFHR：左コン位置，LFHR：左コン回転，RGHP：右コン位置，RGHR：右コン回転）
        /*Debug.Log("HMDPosition:" + HMDPosition.x + ", " + HMDPosition.y + ", " + HMDPosition.z + "\n" +
                    "HMDRotation:" + HMDRotation.x + ", " + HMDRotation.y + ", " + HMDRotation.z);*/
        //Debug.Log("HMDRotationQ:" + HMDRotationQ.x + ", " + HMDRotationQ.y + ", " + HMDRotationQ.z + " ," + HMDRotationQ.w);
        hmdRotationText.text = "x : "+HMDRotation.x + ", y ;  " + HMDRotation.y + ", z : " + HMDRotation.z;
        hmdPosText.text = "(" + HMDPosition.x + ", " + HMDPosition.y + ", " + HMDPosition.z + ")";


    }

    public Vector3 getHMDPos()
    {
        return HMDPosition;
    }

    public Vector3 getHMDRotation()
    {
        return HMDRotation;
    }
}