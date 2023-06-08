using System;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.UI;

public class HMDDataCapture : MonoBehaviour
{
    //HMD�̈ʒu���W�i�[�p
    private Vector3 HMDPosition;
    //HMD�̉�]���W�i�[�p�i�N�H�[�^�j�I���j
    private Quaternion HMDRotationQ;
    //HMD�̉�]���W�i�[�p�i�I�C���[�p�j
    private Vector3 HMDRotation;
    [SerializeField] private GameObject hmdRotationTextObject;
    [SerializeField] private GameObject hmdPosTextObject;
    private Text hmdRotationText;
    private Text hmdPosText;

    //1�t���[�����ɌĂяo�����Update���]�b�g
    private void Start()
    {
        hmdRotationText = hmdRotationTextObject.GetComponent<Text>();
        hmdPosText = hmdPosTextObject.GetComponent<Text>();
    }
    void Update()
    {

        /*InputTracking.GetLocalPosition(XRNode.�@�햼)�ŋ@��̈ʒu��������Ăяo����*/

        //Head�i�w�b�h�}�E���h�f�B�X�v���C�j�̏����ꎞ�ۊ�-----------
        //�ʒu���W���擾
        HMDPosition = InputTracking.GetLocalPosition(XRNode.Head);
        //��]���W���N�H�[�^�j�I���Œl���󂯎��
        HMDRotationQ = InputTracking.GetLocalRotation(XRNode.Head);
        //�擾�����l���N�H�[�^�j�I�� �� �I�C���[�p�ɕϊ�
        HMDRotation = HMDRotationQ.eulerAngles;
        //--------------------------------------------------------------



        //�擾�����f�[�^��\���iHMDP�FHMD�ʒu�CHMDR�FHMD��]�CLFHR�F���R���ʒu�CLFHR�F���R����]�CRGHP�F�E�R���ʒu�CRGHR�F�E�R����]�j
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