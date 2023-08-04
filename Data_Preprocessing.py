#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import os
current_path = os.getcwd()
current_path


# In[3]:


df0=pd.read_csv('DiagnosisEventFact.csv')
df1=pd.read_csv('EdVisitFact.csv')
df2=pd.read_csv('EncounterFact.csv')
df3=pd.read_csv('HospitalAdmissionFact.csv')
df4=pd.read_csv('LabComponentResultFact.csv')
df5=pd.read_csv('MedicationOrderFact.csv')
df6=pd.read_csv('ProcedureOrderFact.csv')
df7=pd.read_csv('PatientDim.csv')


# In[4]:


df0.columns = ["PatientDurableKey", "DiagnosisName", "DepartmentName"]


# In[5]:


df0['DiagnosisName'] = df0['DiagnosisName'].astype(str)+'_DX'
df0['DepartmentName'] = df0['DepartmentName'].astype(str)+'_DPT'


# In[6]:


df0


# In[7]:


new = df0[['PatientDurableKey', 'DiagnosisName']].copy()
new1 = df0[['PatientDurableKey', 'DepartmentName']].copy()


# In[8]:


new.rename(columns={'DiagnosisName': 'Feature'}, inplace=True)
new1.rename(columns={'DepartmentName': 'Feature'}, inplace=True)


# In[9]:


dfq=new.groupby(["PatientDurableKey","Feature"]).size().reset_index(name="DepFreq")
dfw=new1.groupby(["PatientDurableKey","Feature"]).size().reset_index(name="DepFreq")


# In[10]:


dfq


# In[11]:


df_final=dfq.append(dfw, ignore_index=True)


# In[12]:


df_final


# In[13]:


df_final=df_final[~df_final.Feature.str.contains('\*Not Applicable_')]
df_final=df_final[~df_final.Feature.str.contains('\*Unspecified_')]


# In[14]:


diagnosis=df_final


# In[15]:


diagnosis.sort_values('DepFreq',ascending=False)


# In[16]:


df1.columns = ["PatientDurableKey", "PrimaryEdDiagnosisName", "PrimaryChiefComplaintName"]


# In[17]:


df1


# In[18]:


df1['PrimaryEdDiagnosisName'] = df1['PrimaryEdDiagnosisName'].astype(str)+'_DX'
df1['PrimaryChiefComplaintName'] = df1['PrimaryChiefComplaintName'].astype(str)+'_CCOMP'


# In[19]:


df1


# In[20]:


new = df1[['PatientDurableKey', 'PrimaryEdDiagnosisName']].copy()
new1 = df1[['PatientDurableKey', 'PrimaryChiefComplaintName']].copy()
new.rename(columns={'PrimaryEdDiagnosisName': 'Feature'}, inplace=True)
new1.rename(columns={'PrimaryChiefComplaintName': 'Feature'}, inplace=True)
dfq=new.groupby(["PatientDurableKey","Feature"]).size().reset_index(name="DepFreq")
dfw=new1.groupby(["PatientDurableKey","Feature"]).size().reset_index(name="DepFreq")
df_final=dfq.append(dfw, ignore_index=True)
df_final=df_final[~df_final.Feature.str.contains('\*Not Applicable_')]
df_final=df_final[~df_final.Feature.str.contains('\*Unspecified_')]
edvisit=df_final


# In[21]:


edvisit.sort_values('DepFreq',ascending=False)


# In[22]:


df2.columns = ["PatientDurableKey", "PrimaryDiagnosisKey", "PrimaryDiagnosisName", "PrimaryProcedureKey", "PrimaryProcedureName","PrimaryProcedureCategory","PrimaryProcedureCptCode","PrimaryProcedureHcpcsCode" ]
df2['PrimaryDiagnosisName'] = df2['PrimaryDiagnosisName'].astype(str)+'_DX'
df2['PrimaryProcedureName'] = df2['PrimaryProcedureName'].astype(str)+'_PROC'

new = df2[['PatientDurableKey', 'PrimaryDiagnosisName']].copy()
new1 = df2[['PatientDurableKey', 'PrimaryProcedureName']].copy()
new.rename(columns={'PrimaryDiagnosisName': 'Feature'}, inplace=True)
new1.rename(columns={'PrimaryProcedureName': 'Feature'}, inplace=True)
dfq=new.groupby(["PatientDurableKey","Feature"]).size().reset_index(name="DepFreq")
dfw=new1.groupby(["PatientDurableKey","Feature"]).size().reset_index(name="DepFreq")
df_final=dfq.append(dfw, ignore_index=True)
df_final=df_final[~df_final.Feature.str.contains('\*Not Applicable_')]
df_final=df_final[~df_final.Feature.str.contains('\*Unspecified_')]
encounter=df_final


# In[23]:


encounter.sort_values('DepFreq',ascending=False)


# In[24]:


df3.columns = ["PatientDurableKey", "PrincipalProblemDiagnosisName", "PrimaryCodedDiagnosisName", "PrimaryCodedProcedureName" ]
df3['PrincipalProblemDiagnosisName'] = df3['PrincipalProblemDiagnosisName'].astype(str)+'_DX'
df3['PrimaryCodedProcedureName'] = df3['PrimaryCodedProcedureName'].astype(str)+'_PROC'

new = df3[['PatientDurableKey', 'PrincipalProblemDiagnosisName']].copy()
new1 = df3[['PatientDurableKey', 'PrimaryCodedProcedureName']].copy()
new.rename(columns={'PrincipalProblemDiagnosisName': 'Feature'}, inplace=True)
new1.rename(columns={'PrimaryCodedProcedureName': 'Feature'}, inplace=True)
dfq=new.groupby(["PatientDurableKey","Feature"]).size().reset_index(name="DepFreq")
dfw=new1.groupby(["PatientDurableKey","Feature"]).size().reset_index(name="DepFreq")
df_final=dfq.append(dfw, ignore_index=True)
df_final=df_final[~df_final.Feature.str.contains('\*Not Applicable_')]
df_final=df_final[~df_final.Feature.str.contains('\*Unspecified_')]
hospital=df_final
hospital.sort_values('DepFreq',ascending=False)


# In[25]:


df4.columns = ["PatientDurableKey", "ComponentName", "ProcedureName", "ProcedureCptCode" ]
df4['ComponentName'] = df4['ComponentName'].astype(str)+'_CN'
df4['ProcedureName'] = df4['ProcedureName'].astype(str)+'_LPROC'

new = df4[['PatientDurableKey', 'ComponentName']].copy()
new1 = df4[['PatientDurableKey', 'ProcedureName']].copy()
new.rename(columns={'ComponentName': 'Feature'}, inplace=True)
new1.rename(columns={'ProcedureName': 'Feature'}, inplace=True)
dfq=new.groupby(["PatientDurableKey","Feature"]).size().reset_index(name="DepFreq")
dfw=new1.groupby(["PatientDurableKey","Feature"]).size().reset_index(name="DepFreq")
df_final=dfq.append(dfw, ignore_index=True)
df_final=df_final[~df_final.Feature.str.contains('\*Not Applicable_')]
df_final=df_final[~df_final.Feature.str.contains('\*Unspecified_')]
lab=df_final
lab.sort_values('DepFreq',ascending=False)


# In[26]:


df5.columns = ["PatientDurableKey", "MedicationName", "MedicationTherapeuticClass", "MedicationPharmaceuticalClass" ]
df5['MedicationName'] = df5['MedicationName'].astype(str)+'_MED'
df5['MedicationTherapeuticClass'] = df5['MedicationTherapeuticClass'].astype(str)+'_MTC'
df5['MedicationPharmaceuticalClass'] = df5['MedicationPharmaceuticalClass'].astype(str)+'_MPC'

new = df5[['PatientDurableKey', 'MedicationName']].copy()
new1 = df5[['PatientDurableKey', 'MedicationTherapeuticClass']].copy()
new2 = df5[['PatientDurableKey', 'MedicationPharmaceuticalClass']].copy()
new.rename(columns={'MedicationName': 'Feature'}, inplace=True)
new1.rename(columns={'MedicationTherapeuticClass': 'Feature'}, inplace=True)
new2.rename(columns={'MedicationPharmaceuticalClass': 'Feature'}, inplace=True)
dfq=new.groupby(["PatientDurableKey","Feature"]).size().reset_index(name="DepFreq")
dfw=new1.groupby(["PatientDurableKey","Feature"]).size().reset_index(name="DepFreq")
dfx=new2.groupby(["PatientDurableKey","Feature"]).size().reset_index(name="DepFreq")
df_final=dfq.append(dfw, ignore_index=True)
df_final=df_final.append(dfx, ignore_index=True)
df_final=df_final[~df_final.Feature.str.contains('\*Not Applicable_')]
df_final=df_final[~df_final.Feature.str.contains('\*Unspecified_')]
medication=df_final
medication.sort_values('DepFreq',ascending=False)


# In[27]:


df6.columns = ["PatientDurableKey", "ProcedureName", "ProcedureCategory" ]
df6['ProcedureName'] = df6['ProcedureName'].astype(str)+'_PROC'


new = df6[['PatientDurableKey', 'ProcedureName']].copy()
new.rename(columns={'ProcedureName': 'Feature'}, inplace=True)
dfq=new.groupby(["PatientDurableKey","Feature"]).size().reset_index(name="DepFreq")
df_final=dfq
df_final=df_final[~df_final.Feature.str.contains('\*Not Applicable_')]
df_final=df_final[~df_final.Feature.str.contains('\*Unspecified_')]
procedure=df_final
procedure.sort_values('DepFreq',ascending=False)


# In[28]:


df7.columns = ["PatientDurableKey", "Sex", "Ethnicity","FirstRace","BirthDate","SmokingStatus" ]
df7['Sex'] = df7['Sex'].astype(str)+'_DM'


new = df7[['PatientDurableKey', 'Sex']].copy()
new.rename(columns={'Sex': 'Feature'}, inplace=True)
dfq=new.groupby(["PatientDurableKey","Feature"]).size().reset_index(name="DepFreq")
df_final=dfq
df_final=df_final[~df_final.Feature.str.contains('\*Not Applicable_')]
df_final=df_final[~df_final.Feature.str.contains('\*Unspecified_')]
demographics=df_final
demographics.sort_values('DepFreq',ascending=False)


# In[29]:


df_complete=pd.concat([diagnosis,edvisit,encounter,hospital,lab,medication, procedure,demographics],ignore_index=False)


# In[30]:


df_complete


# In[31]:


df_complete=df_complete.drop_duplicates(['PatientDurableKey', 'Feature'])
df_complete=df_complete[df_complete.PatientDurableKey!=-1]


# In[32]:


dfq = df_complete.set_index(['PatientDurableKey','Feature'])['DepFreq'].unstack()


# In[33]:


dfq=dfq.fillna(0.01)
dfq['Label']=0
dfq


# In[36]:


dfq.to_csv('PatientSnapshot.csv')


# In[34]:


dfq=dfq.iloc[1:]


# In[35]:


dfq


# In[ ]:




