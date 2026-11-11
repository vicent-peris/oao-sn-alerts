import os
import csv
import json
import requests
import numpy as np
from datetime import datetime, timezone
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from astroplan import Observer
from alerce.core import Alerce

alerce_client = Alerce()


TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

TNS_API_KEY = os.environ.get("TNS_API_KEY", "")
TNS_BOT_ID = os.environ.get("TNS_BOT_ID", "")
TNS_BOT_NAME = os.environ.get("TNS_BOT_NAME", "TURIA2_Bot")

OAO_LOC = EarthLocation(lat=39.945 * u.deg, lon=-1.102 * u.deg, height=1280 * u.m)
OBSERVATORY = Observer(location=OAO_LOC, name="OAO-TURIA2", timezone="Europe/Madrid")

CONFIG_CSV = "config_exposiciones.csv"
DB_FILE = "supernovas_seguimiento.json"
DIAS_MAX_ACTIVAS = 125.0

SNR_OBJETIVO = {
    "CaK": 35.0, "OIII": 35.0, "SII": 35.0, "Ha": 25.0,
    "u": 20.0, "g": 30.0, "r": 30.0, "i": 30.0
}

def cargar_config_exposiciones():
    config = {}
    with open(CONFIG_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filtro = row["filtro"].strip()
            config[filtro] = {
                "mag_ref": float(row["magnitud"]),
                "snr_ref": float(row["SNR"]),
                "t_ref": float(row["tiempo_segundos"])
            }
    return config

def calcular_tiempos_exposicion(mag_actual, config_csv):
    tiempos_seg = {}
    for filtro, ref in config_csv.items():
        snr_req = SNR_OBJETIVO.get(filtro, ref["snr_ref"])
        factor_snr = (snr_req / ref["snr_ref"]) ** 2
        factor_flujo = 10 ** (0.8 * (mag_actual - ref["mag_ref"]))
        t_calc = ref["t_ref"] * factor_snr * factor_flujo
        tiempos_seg[filtro] = max(30.0, min(18000.0, t_calc))
    return tiempos_seg

def cargar_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "supernovas" not in data:
                return {"telegram_offset": 0, "supernovas": data}
            return data
    return {"telegram_offset": 0, "supernovas": {}}

def guardar_db(db):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4)
def evaluar_visibilidad_oao(ra_deg, dec_deg):
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    ahora = Time(datetime.now(timezone.utc))

    ocaso = OBSERVATORY.twilight_evening_astronomical(ahora, which="next")
    orto = OBSERVATORY.twilight_morning_astronomical(ocaso, which="next")

    duracion_noche_h = (orto - ocaso).to(u.hour).value
    pasos = int(duracion_noche_h * 6)
    if pasos <= 0:
        return 0.0, None, None

    tiempos = ocaso + np.linspace(0, duracion_noche_h, pasos) * u.hour
    altaz = target.transform_to(AltAz(obstime=tiempos, location=OAO_LOC))

    indices_utiles = np.where(altaz.alt >= 30.0 * u.deg)[0]
    if len(indices_utiles) == 0:
        return 0.0, None, None

    horas_utiles = len(indices_utiles) * (10.0 / 60.0)
    t_ini = tiempos[indices_utiles[0]].to_datetime(timezone=OBSERVATORY.timezone).strftime("%H:%M")
    t_fin = tiempos[indices_utiles[-1]].to_datetime(timezone=OBSERVATORY.timezone).strftime("%H:%M")
    return horas_utiles, t_ini, t_fin

def obtener_datos_alerce_detallados(oid):
    try:
        obj_df = alerce_client.query_objects(oid=oid, format="pandas")
        if obj_df.empty:
            return None, [], None
        obj_data = obj_df.iloc[0].to_dict()

        det_df = alerce_client.query_detections(oid=oid, format="pandas")
        mapa_filtros_ztf = {1: "ZTF-g", 2: "ZTF-r", 3: "ZTF-i"}
        if not det_df.empty and "fid" in det_df.columns:
            filtros_obs = sorted(list({mapa_filtros_ztf.get(f, "ZTF-?") for f in det_df["fid"].dropna()}))
        else:
            filtros_obs = []
    except Exception as e:
        print(f"Error consultando detalles de {oid}: {e}")
        return None, [], None

    first_mjd = obj_data.get("firstmjd")
    dias_desde_explosion = None
    if first_mjd is not None and not np.isnan(first_mjd):
        ahora_mjd = Time(datetime.now(timezone.utc)).mjd
        dias_desde_explosion = max(0.1, ahora_mjd - float(first_mjd))

    return obj_data, filtros_obs, dias_desde_explosion

def consultar_tns_espectros_y_filtros(ra_deg, dec_deg):
    if not TNS_API_KEY:
        return "Sin API TNS", 0, []

    url_search = "https://wis-tns.org"
    headers = {"User-Agent": f'tns_marker{{"tns_id":{TNS_BOT_ID},"type":"bot","name":"{TNS_BOT_NAME}"}}'}
    search_data = {"ra": str(ra_deg), "dec": str(dec_deg), "radius": "5", "units": "arcsec"}

    try:
        res = requests.post(url_search, headers=headers, data={"api_key": TNS_API_KEY, "data": json.dumps(search_data)}, timeout=10)
        items = res.json().get("data", {}).get("reply", [])
        if not items:
            return "No catalogada en TNS", 0, []

        nombre_iau = items[0].get("objname")
        url_obj = "https://wis-tns.org"
        obj_req = {"objname": nombre_iau, "photometry": "1", "spectra": "1"}
        res_obj = requests.post(url_obj, headers=headers, data={"api_key": TNS_API_KEY, "data": json.dumps(obj_req)}, timeout=10)
        reply = res_obj.json().get("data", {}).get("reply", {})

        espectros = reply.get("spectra", [])
        fotometria = reply.get("photometry", [])
        filtros_tns = sorted(list({p.get("filters", {}).get("name", "") for p in fotometria if p.get("filters")}))
        return f"SN {nombre_iau}", len(espectros), filtros_tns
    except Exception:
        return "Error consulta TNS", 0, []
def enviar_mensaje(texto):
    url = f"https://telegram.org{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": texto, "parse_mode": "Markdown"})

def revisar_comandos_pendientes(db):
    offset = db.get("telegram_offset", 0)
    url = f"https://telegram.org{TELEGRAM_TOKEN}/getUpdates"
    try:
        res = requests.get(url, params={"offset": offset, "timeout": 5}, timeout=10).json()
    except Exception:
        return

    pedir_activas = False
    pedir_estado = False

    for update in res.get("result", []):
        db["telegram_offset"] = update["update_id"] + 1
        msg = update.get("message", {})
        texto = msg.get("text", "").strip().lower()
        if texto.startswith("/activas"):
            pedir_activas = True
        elif texto.startswith("/estado"):
            pedir_estado = True

    if pedir_activas:
        responder_comando_activas(db)
    if pedir_estado:
        responder_comando_estado(db)

def responder_comando_activas(db):
    ahora_mjd = Time(datetime.now(timezone.utc)).mjd
    activas = [sn for sn in db["supernovas"].values() if sn.get("activa", False)]
    if not activas:
        enviar_mensaje("📭 No hay supernovas activas en seguimiento en este momento.")
        return

    lineas = ["📋 *SUPERNOVAS ACTIVAS EN SEGUIMIENTO PAI*\n"]
    for sn in activas:
        edad = ahora_mjd - sn["firstmjd"]
        dias_restantes = max(0, DIAS_MAX_ACTIVAS - edad)
        lineas.append(f"• `{sn['oid']}` ({sn['nombre_tns']}): *Día {edad:.1f}* (Quedan {dias_restantes:.0f}d)")
    enviar_mensaje("\n".join(lineas))

def responder_comando_estado(db):
    ahora_mjd = Time(datetime.now(timezone.utc)).mjd
    activas = [sn for sn in db["supernovas"].values() if sn.get("activa", False)]
    if not activas:
        enviar_mensaje("📭 No hay supernovas activas para actualizar su estado.")
        return

    lineas = ["📊 *ESTADO ACTUALIZADO DE CAMPAÑA*\n"]
    for sn in activas:
        oid = sn["oid"]
        edad = ahora_mjd - sn["firstmjd"]
        _, filtros_ztf, _ = obtener_datos_alerce_detallados(oid)
        nombre_tns, num_esp_nuevo, filtros_tns = consultar_tns_espectros_y_filtros(sn["ra"], sn["dec"])

        if nombre_tns != "No catalogada en TNS":
            sn["nombre_tns"] = nombre_tns

        filtros_actualizados = sorted(list(set(sn["filtros_obs"] + filtros_ztf + filtros_tns)))
        nuevos_filtros = set(filtros_actualizados) - set(sn["filtros_obs"])

        aviso_esp = ""
        if num_esp_nuevo > sn["num_espectros"]:
            aviso_esp = f" 🆕 *(+{num_esp_nuevo - sn['num_espectros']} espectro nuevo!)*"
            sn["num_espectros"] = num_esp_nuevo

        aviso_filtros = f" (Nuevos: {', '.join(nuevos_filtros)})" if nuevos_filtros else ""
        sn["filtros_obs"] = filtros_actualizados

        lineas.append(
            f"🔭 *{sn['oid']}* ({sn['nombre_tns']}) - *Día {edad:.1f}*\n"
            f"   🧬 Espectros: {sn['num_espectros']}{aviso_esp}\n"
            f"   🎨 Filtros: {', '.join(filtros_actualizados)}{aviso_filtros}\n"
        )
    enviar_mensaje("\n".join(lineas))
def procesar_alertas(db):
    config_csv = cargar_config_exposiciones()
    ahora_mjd = Time(datetime.now(timezone.utc)).mjd

    for oid, info in db["supernovas"].items():
        if info.get("activa", False) and (ahora_mjd - info["firstmjd"]) > DIAS_MAX_ACTIVAS:
            info["activa"] = False

    try:
        df_cand = alerce_client.query_objects(
            classifier="lc_classifier",
            class_name="SN",
            probability=0.6,
            firstmjd=[ahora_mjd - 5.0, ahora_mjd],
            page_size=50,
            order_by="lastmjd",
            order_mode="DESC",
            format="pandas"
        )
        candidatos = df_cand.to_dict(orient="records") if not df_cand.empty else []
        print(f"Candidatos encontrados en ALeRCE: {len(candidatos)}")
    except Exception as e:
        print(f"Error consultando lista en ALeRCE: {e}")
        return

    for obj in candidatos:
        oid = obj.get("oid")
        if oid in db["supernovas"]:
            continue

        ra, dec = obj.get("meanra"), obj.get("meandec")
        if ra is None or dec is None or dec <= -10.0:
            continue

        detalles, filtros_ztf, edad_dias = obtener_datos_alerce_detallados(oid)
        if detalles is None:
            continue

        if edad_dias is not None and edad_dias > 5.0:
            continue

        mags_validas = [m for m in (obj.get("last_g"), obj.get("last_r")) if m is not None]
        mag_actual = min(mags_validas) if mags_validas else None
        if mag_actual is not None and mag_actual > 17.5:
            continue

        ebv = detalles.get("ebv")
        if ebv is not None and ebv >= 0.25:
            continue

        redshift = detalles.get("z")
        if redshift is not None and (redshift > 0.027 or redshift <= 0):
            continue

        horas_utiles, t_ini, t_fin = evaluar_visibilidad_oao(ra, dec)
        if horas_utiles < 2.0:
            continue

        mag_calculo = mag_actual if mag_actual is not None else 16.5
        tiempos_seg = calcular_tiempos_exposicion(mag_calculo, config_csv)

        nombre_tns, num_espectros, filtros_tns = consultar_tns_espectros_y_filtros(ra, dec)
        todos_filtros = sorted(list(set(filtros_ztf + filtros_tns)))

        db["supernovas"][oid] = {
            "oid": oid, "nombre_tns": nombre_tns, "ra": ra, "dec": dec,
            "firstmjd": detalles.get("firstmjd", ahora_mjd), "mag": mag_actual,
            "redshift": redshift, "ebv": ebv, "num_espectros": num_espectros,
            "filtros_obs": todos_filtros, "activa": True
        }

        enviar_ficha_telegram(db["supernovas"][oid], edad_dias, horas_utiles, f"{t_ini}-{t_fin}", tiempos_seg)

def enviar_ficha_telegram(sn, edad_dias, horas_utiles, ventana, tiempos_seg):
    coord_str = SkyCoord(ra=sn["ra"] * u.deg, dec=sn["dec"] * u.deg).to_string("hmsdms", sep=":")
    edad_str = f"{edad_dias:.1f} días" if edad_dias is not None else "Desconocida (Dato nulo)"
    mag_str = f"{sn['mag']:.2f} mag" if sn["mag"] is not None else "Sin dato (Calculado a 16.5)"
    dist_str = f"{(sn['redshift'] * 299792.458 / 70.0):.1f} Mpc" if sn["redshift"] else "Desconocida"
    ebv_str = f"{sn['ebv']:.3f}" if sn["ebv"] is not None else "Desconocido"
    esp_str = f"Sí ({sn['num_espectros']} en TNS)" if sn["num_espectros"] > 0 else "No todavía"
    filtros_str = ", ".join(sn["filtros_obs"]) if sn["filtros_obs"] else "Ninguno registrado"

    t_cak = tiempos_seg["CaK"] / 60.0
    t_u = tiempos_seg["u"] / 60.0
    t_g = tiempos_seg["g"] / 60.0
    t_sho = tiempos_seg["OIII"] / 60.0

    mensaje = (
        f"🚨 *ALERTA SUPERNOVA - OAO (TURIA-2)* 🚨\n\n"
        f"🔭 *ID:* `{sn['oid']}` ({sn['nombre_tns']})\n"
        f"📍 *Coord (J2000):* `{coord_str}`\n"
        f"⏳ *Tiempo desde explosión:* *{edad_str}*\n"
        f"✨ *Magnitud:* {mag_str}\n"
        f"🌌 *Distancia est.:* {dist_str} | *Polvo MW:* {ebv_str}\n"
        f"🧬 *Espectro público:* {esp_str}\n"
        f"🎨 *Filtros ya observados:* {filtros_str}\n\n"
        f"⏱️ *Visibilidad OAO (>30°):* *{horas_utiles:.1f} h* ({ventana} CET)\n"
        f"📸 *Tiempos guía (según CSV):*\n"
        f"   • Prioridad temprana: *u:* {t_u:.0f} min | *g:* {t_g:.0f} min | *CaK:* {t_cak:.0f} min\n"
        f"   • Bandas estrechas (3nm): *SHO:* {t_sho:.0f} min c/u\n\n"
        f"🔗 [Ficha en ALeRCE](https://alerce.online{sn['oid']})"
    )
    enviar_mensaje(mensaje)

if __name__ == "__main__":
    db = cargar_db()
    revisar_comandos_pendientes(db)
    procesar_alertas(db)
    guardar_db(db)
