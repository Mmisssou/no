#!/usr/bin/env python
# coding: utf-8

# In[227]:



import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import shap
import time
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner





st.set_page_config(page_title='Scoring  des demandes de prêt',
                    page_icon='random',
                    layout='centered',
                    initial_sidebar_state='auto')

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 2


def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Home", "Project 7 OC", "Contact"],  # required
                icons=["house", "book", "envelope"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["Home", "Project7 OC", "Contact"],  # required
            icons=["house", "book", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "gold"},
            },
        )
        return selected

selected = streamlit_menu(example=EXAMPLE_NO)





# Display the title
st.title('Scoring  des demandes de prêt')
st.header("Projet N°: 07  -encadré par :El hadji Abdoulaye Thiam- ")
path = "scoring.jpeg"
image = Image.open(path)
st.sidebar.image(image, width=180)
st.sidebar.header('Variables dentrée utilisateur')


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url_hello = "https://assets4.lottiefiles.com/packages/lf20_gb33dzsv.json"
lottie_hello = load_lottieurl(lottie_url_hello)
st_lottie(lottie_hello, key="hello")




API_URL = "https://loan-heroku.herokuapp.com/app/"


# In[230]:


def get_list_display_features(f, def_n, key):
        all_feat = f
        n = st.slider("Nb de variables à afficher",
                      min_value=2, max_value=40,
                      value=def_n, step=None, format=None, key=key)

        disp_cols = list(get_features_importances().sort_values(ascending=False).iloc[:n].index)

        box_cols = st.multiselect(
            'Choisissez les variables à afficher:',
            sorted(all_feat),
            default=disp_cols, key=key)
        return box_cols


# In[231]:

@st.cache(suppress_st_warning=True)
def get_id_list():
        # URL of the sk_id API
        id_api_url = API_URL + "id/"
        # Requesting the API and saving the response
        response = requests.get(id_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        id_customers = pd.Series(content['data']).values
        return id_customers


# In[232]:


data_type = []


# In[233]:

@st.cache
def get_selected_cust_data(selected_id):
        # URL of the sk_id API
        data_api_url = API_URL + "data_cust/?SK_ID_CURR=" + str(selected_id)
        # Requesting the API and saving the response
        response = requests.get(data_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        x_custom = pd.DataFrame(content['data'])
        # x_cust = json_normalize(content['data'])
        y_customer = (pd.Series(content['y_cust']).rename('TARGET'))
        # y_customer = json_normalize(content['y_cust'].rename('TARGET'))
        return x_custom, y_customer


# In[234]:

@st.cache
def get_all_cust_data():
        # URL of the sk_id API
        data_api_url = API_URL + "all_proc_data_tr/"
        # Requesting the API and saving the response
        response = requests.get(data_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))  #
        x_all_cust = json_normalize(content['X_train'])  # Results contain the required data
        y_all_cust = json_normalize(content['y_train'].rename('TARGET'))  # Results contain the required data
        return x_all_cust, y_all_cust


# In[235]:

@st.cache
def get_score_model(selected_id):
       # URL of the sk_id API
       score_api_url = API_URL + "scoring_cust/?SK_ID_CURR=" + str(selected_id)
       # Requesting the API and saving the response
       response = requests.get(score_api_url)
       # Convert from JSON format to Python dict
       content = json.loads(response.content.decode('utf-8'))
       # Getting the values of "ID" from the content
       score_model = (content['score'])
       threshold = content['thresh']
       return score_model, threshold


# In[236]:

@st.cache
def values_shap(selected_id):
        # URL of the sk_id API
        shap_values_api_url = API_URL + "shap_val/?SK_ID_CURR=" + str(selected_id)
        # Requesting the API and saving the response
        response = requests.get(shap_values_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        shapvals = pd.DataFrame(content['shap_val_cust'].values())
        expec_vals = pd.DataFrame(content['expected_vals'].values())
        return shapvals, expec_vals


# In[237]:

@st.cache
def values_expect():
       # URL of the sk_id API
       expected_values_api_url = API_URL + "exp_val/"
       # Requesting the API and saving the response
       response = requests.get(expected_values_api_url)
       # Convert from JSON format to Python dict
       content = json.loads(response.content)
       # Getting the values of "ID" from the content
       expect_vals = pd.Series(content['data']).values
       return expect_vals


# In[238]:

@st.cache
def feat():
        # URL of the sk_id API
        feat_api_url = API_URL + "feat/"
        # Requesting the API and saving the response
        response = requests.get(feat_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        features_name = pd.Series(content['data']).values
        return features_name


# In[239]:

@st.cache
def get_features_importances():
        # URL of the aggregations API
        feat_imp_api_url = API_URL + "feat_imp/"
        # Requesting the API and save the response
        response = requests.get(feat_imp_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert back to pd.Series
        feat_imp = pd.Series(content['data']).sort_values(ascending=False)
        return feat_imp


# In[240]:

@st.cache
def get_data_neigh(selected_id):
       # URL of the scoring API (ex: SK_ID_CURR = 100005)
       neight_data_api_url = API_URL + "neigh_cust/?SK_ID_CURR=" + str(selected_id)
       # save the response of API request
       response = requests.get(neight_data_api_url)
       # convert from JSON format to Python dict
       content = json.loads(response.content.decode('utf-8'))
       # convert data to pd.DataFrame and pd.Series
       # targ_all_cust = (pd.Series(content['target_all_cust']).rename('TARGET'))
       # target_select_cust = (pd.Series(content['target_selected_cust']).rename('TARGET'))
       # data_all_customers = pd.DataFrame(content['data_all_cust'])
       data_neig = pd.DataFrame(content['data_neigh'])
       target_neig = (pd.Series(content['y_neigh']).rename('TARGET'))
       return data_neig, target_neig


# In[241]:


def get_data_thousand_neigh(selected_id):
       thousand_neight_data_api_url = API_URL + "thousand_neigh/?SK_ID_CURR=" + str(selected_id)
       # save the response of API request
       response = requests.get(thousand_neight_data_api_url)
       # convert from JSON format to Python dict
       content = json.loads(response.content.decode('utf-8'))
       # convert data to pd.DataFrame and pd.Series
       # targ_all_cust = (pd.Series(content['target_all_cust']).rename('TARGET'))
       # target_select_cust = (pd.Series(content['target_selected_cust']).rename('TARGET'))
       # data_all_customers = pd.DataFrame(content['data_all_cust'])
       data_thousand_neig = pd.DataFrame(content['X_thousand_neigh'])
       x_custo = pd.DataFrame(content['x_custom'])
       target_thousand_neig = (pd.Series(content['y_thousand_neigh']).rename('TARGET'))
       return data_thousand_neig, target_thousand_neig, x_custo


# In[242]:

@st.cache
def get_shap_values(select_sk_id):
       # URL of the scoring API
       GET_SHAP_VAL_API_URL = API_URL + "shap_values/?SK_ID_CURR=" + str(select_sk_id)
       # save the response of API request
       response = requests.get(GET_SHAP_VAL_API_URL)
       # convert from JSON format to Python dict
       content = json.loads(response.content.decode('utf-8'))
       # convert data to pd.DataFrame or pd.Series
       shap_val_df = pd.DataFrame(content['shap_val'])
       shap_val_trans = pd.Series(content['shap_val_cust_trans'])
       exp_value = content['exp_val']
       exp_value_trans = content['exp_val_trans']
       X_neigh_ = pd.DataFrame(content['X_neigh_'])
       return shap_val_df, shap_val_trans, exp_value, exp_value_trans, X_neigh_


# In[243]:


# list of customer's ID's
cust_id = get_id_list()
# Selected customer's ID
selected_id = st.sidebar.selectbox('Sélectionnez le ID client dans la liste:', cust_id, key=18)
st.write('Votre identifiant sélectionné= ', selected_id)


# In[244]:


############################################################################
 #                           Graphics Functions
 ############################################################################


# In[245]:

@st.cache
def shap_summary():
    return shap.summary_plot(shap_vals, feature_names=features)


# In[246]:

@st.cache
def waterfall_plot(nb, ft, expected_val, shap_val):
        return shap.plots._waterfall.waterfall_legacy(expected_val, shap_val[0, :],
                                                      max_display=nb, feature_names=ft)


# In[247]:

@st.cache(allow_output_mutation=True) 
def force_plot():
        shap.initjs()
        return shap.force_plot(expected_vals[0][0], shap_vals[0, :], matplotlib=True)


# In[248]:

@st.cache
def gauge_plot(scor, th):
        scor = int(scor * 100)
        th = int(th * 100)

        if scor >= th:
            couleur_delta = 'golden'
        elif scor < th:
            couleur_delta = 'Orange'

        if scor >= th:
            valeur_delta = "blue"
        elif scor < th:
            valeur_delta = "green"

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=scor,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Score client sélectionné", 'font': {'size': 25}},
            delta={'reference': int(th), 'en augmentant': {'color': valeur_delta}},
            gauge={
                'axis': {'range': [None, int(100)], 'tickwidth': 1.5, 'tickcolor': "black"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, int(th)], 'color': 'darkseagreen'},
                    {'range': [int(th), int(scor)], 'color': couleur_delta}],
                'seuil': {
                    'line': {'color': "gold", 'width': 4},
                    'thickness': 1,
                    'value': int(th)}}))

        fig.update_layout(paper_bgcolor="lavender", font={'color': "darkblue", 'family': "Arial"})
        return fig


# In[249]:


if st.sidebar.checkbox("Données du client"):
    st.markdown('données du client sélectionné :')
data_selected_cust, y_cust = get_selected_cust_data(selected_id)
# data_selected_cust.columns = data_selected_cust.columns.str.split('.').str[0]
st.write(data_selected_cust)


# In[250]:


if st.sidebar.checkbox("La décision du modèle", key=38):
       # Get score & threshold model
       score, threshold_model = get_score_model(selected_id)
       # Display score (default probability)
       st.write('Probabilité de défaut : {:.0f}%'.format(score * 100))
       # Display default threshold
       st.write('Seuil de modèle par défaut : {:.0f}%'.format(threshold_model * 100))  #
       # Compute decision according to the best threshold (False= loan accepted, True=loan refused)
       if score >= threshold_model:
           decision = "Prêt refusé"
       else:
           decision = "Prêt accordé"
       st.write("Decision :", decision)


# In[251]:


if st.checkbox('Afficher linterprétation  du waterfall', key=25):
            with st.spinner('Graphiques en waterfall plots SHAP en cours daffichage..... Veuillez patienter......'):
                # Get Shap values for customer & expected values
                shap_vals, expected_vals = values_shap(selected_id)
                # index_cust = customer_ind(selected_id)
                # Get features names
                features = feat()
                # st.write(features)
                nb_features = st.slider("Nombre de features à afficher",
                                        min_value=2,
                                        max_value=50,
                                        value=10,
                                        step=None,
                                        format=None,
                                        key=14)
                # draw the waterfall graph (only for the customer with scaling
                waterfall_plot(nb_features, features, expected_vals[0][0], shap_vals.values)

                plt.gcf()
                st.pyplot(plt.gcf())
                # Add markdown
                st.markdown('_Tracé en SHAP waterfall Plot SHAP pour le client demandeur._')
                # Add details title
                expander = st.expander("Concernant le tracé en SHAP waterfall  plot..")
                # Add explanations
                expander.write("Le waterfall  plot  ci-dessus affiche des explications pour la prédiction individuelle du client candidat. Le bas d'un waterfall  plot commence par la valeur attendue de la sortie du modèle (c'est-à-dire la valeur obtenue si aucune information (caractéristiques) n'a été fournie), puis chaque ligne montre comment la contribution positive (rouge) ou négative (bleue) de chaque caractéristique déplace la valeur de la sortie de modèle attendue sur l'ensemble de données d'arrière-plan vers la sortie de modèle pour cette prédiction.")


# In[252]:


if st.checkbox('afficher la distribution des variables par classe', key=20):
    st.header('Boxplots of the main features')
fig, ax = plt.subplots(figsize=(20, 10))
with st.spinner('Boxplot creation in progress...please wait.....'):
# Get Shap values for customer
  shap_vals, expected_vals = values_shap(selected_id)
# Get features names
features = feat()


# In[253]:


disp_box_cols = get_list_display_features(features, 2, key=45)


# In[254]:


data_neigh, target_neigh = get_data_neigh(selected_id)
data_thousand_neigh, target_thousand_neigh, x_customer = get_data_thousand_neigh(selected_id)

x_cust, y_cust = get_selected_cust_data(selected_id)
x_customer.columns = x_customer.columns.str.split('.').str[0]
# Target impuatation (0 : 'repaid (....), 1 : not repaid (....)
# -------------------------------------------------------------
target_neigh = target_neigh.replace({0: 'remboursé (voisins)',
                                     1: 'non remboursé (voisins)'})
target_thousand_neigh = target_thousand_neigh.replace({0: 'remboursé (voisins)',
                                                       1: 'non remboursé (voisins)'})
y_cust = y_cust.replace({0: 'remboursé (voisins)',
                         1: 'non remboursé (voisins)'})

# y_cust.rename(columns={'10006':'TARGET'}, inplace=True)
# ------------------------------
# Get 1000 neighbors personal data
# ------------------------------
df_thousand_neigh = pd.concat([data_thousand_neigh[disp_box_cols], target_thousand_neigh], axis=1)
df_melt_thousand_neigh = df_thousand_neigh.reset_index()
df_melt_thousand_neigh.columns = ['index'] + list(df_melt_thousand_neigh.columns)[1:]
df_melt_thousand_neigh = df_melt_thousand_neigh.melt(id_vars=['index', 'TARGET'],
                                                     value_vars=disp_box_cols,
                                                     var_name="variables",  # "variables",
                                                     value_name="values")


# In[255]:


sns.boxplot(data=df_melt_thousand_neigh, x='variables', y='values',
                            hue='TARGET', linewidth=1, width=0.4,
                            palette=['tab:green', 'tab:red'], showfliers=False,
                            saturation=0.5, ax=ax)


# In[256]:


df_neigh = pd.concat([data_neigh[disp_box_cols], target_neigh], axis=1)
df_melt_neigh = df_neigh.reset_index()
df_melt_neigh.columns = ['index'] + list(df_melt_neigh.columns)[1:]
df_melt_neigh = df_melt_neigh.melt(id_vars=['index', 'TARGET'],
                                                   value_vars=disp_box_cols,
                                                   var_name="variables",  # "variables",
                                                   value_name="values")


# In[257]:


sns.swarmplot(data=df_melt_neigh, x='variables', y='values', hue='TARGET', linewidth=1,
                              palette=['darkgreen', 'darkred'], marker='o', size=15, edgecolor='k', ax=ax)


# In[258]:


df_selected_cust = pd.concat([x_customer[disp_box_cols], y_cust], axis=1)
# st.write("df_sel_cust :", df_sel_cust)
df_melt_sel_cust = df_selected_cust.reset_index()
df_melt_sel_cust.columns = ['index'] + list(df_melt_sel_cust.columns)[1:]
df_melt_sel_cust = df_melt_sel_cust.melt(id_vars=['index', 'TARGET'],
                                                         value_vars=disp_box_cols,
                                                         var_name="variables",
                                                         value_name="values")


# In[259]:


sns.swarmplot(data=df_melt_sel_cust, x='variables', y='values',
                            linewidth=1, color='y', marker='o', size=20,
                            edgecolor='k', label='applicant customer', ax=ax)


# In[260]:


sns.swarmplot(data=df_melt_sel_cust, x='variables', y='values', linewidth=1, color='y', marker='o', size=20,edgecolor='k', label='applicant customer', ax=ax)
# legend



# In[261]:


st.write(fig)  # st.pyplot(fig) # the same


# In[262]:


st.markdown('_Dispersion des principales caractéristiques pour un échantillon aléatoire, 20 voisins les plus proches et client demandeur_')


# In[263]:


expander = st.expander("Concernant le graphique de dispersion....")


# In[264]:


expander.write("Ces boxplots montrent la dispersion des valeurs de caractéristiques prétraitées utilisées par le modèle pour faire une prédiction. Les boxplots vertes sont pour les clients qui ont remboursé leur prêt et les boxplots rouges sont pour les clients qui ne l'ont pas remboursé. Sur les boxplots sont superposées (marqueurs) les valeurs des caractéristiques pour les 20 plus proches voisins du client demandeur dans ensemble d'entraînement. La couleur des marqueurs indique si ces voisins ont remboursé ou non leur emprunt. Les valeurs du client candidat sont superposées en jaune.")

