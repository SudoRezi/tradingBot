#!/usr/bin/env python3
"""
Session Manager - Auto-Resume Trading Bot
Gestisce continuità del trading tra riavvii del sistema
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

class SessionManager:
    """Gestisce stato sessione e auto-resume del trading bot"""
    
    def __init__(self):
        self.session_file = "session_state.json"
        self.config_file = "institutional_config.json"
        
    def save_session_state(self, trading_active: bool, config: Dict[str, Any], 
                          portfolio_state: Dict[str, Any] = None) -> bool:
        """Salva stato sessione corrente"""
        try:
            session_data = {
                "was_active": trading_active,
                "last_active": datetime.now().isoformat(),
                "trading_pairs": config.get("trading_pairs", []),
                "ai_mode": config.get("ai_auto_management", False),
                "risk_level": config.get("risk_level", "moderate"),
                "selected_exchanges": config.get("exchanges", []),
                "portfolio_state": portfolio_state or {},
                "auto_resume_enabled": True,
                "session_id": f"session_{int(datetime.now().timestamp())}"
            }
            
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Errore salvataggio sessione: {e}")
            return False
    
    def load_previous_session(self) -> Optional[Dict[str, Any]]:
        """Carica stato sessione precedente se disponibile"""
        try:
            if not os.path.exists(self.session_file):
                return None
                
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)
            
            # Verifica se la sessione è valida (non troppo vecchia)
            last_active = datetime.fromisoformat(session_data.get("last_active", ""))
            if datetime.now() - last_active > timedelta(days=7):
                # Sessione troppo vecchia, ignora
                return None
            
            return session_data
            
        except Exception as e:
            print(f"Errore caricamento sessione: {e}")
            return None
    
    def should_auto_resume(self) -> tuple[bool, Dict[str, Any]]:
        """Determina se dovrebbe riprendere automaticamente"""
        session = self.load_previous_session()
        
        if not session:
            return False, {}
            
        was_active = session.get("was_active", False)
        auto_resume = session.get("auto_resume_enabled", True)
        
        return was_active and auto_resume, session
    
    def get_resume_summary(self, session: Dict[str, Any]) -> str:
        """Genera summary per UI di resume"""
        if not session:
            return "Nessuna sessione precedente trovata"
        
        last_active = datetime.fromisoformat(session.get("last_active", ""))
        time_diff = datetime.now() - last_active
        
        if time_diff.total_seconds() < 3600:  # Meno di 1 ora
            time_str = f"{int(time_diff.total_seconds() / 60)} minuti fa"
        elif time_diff.total_seconds() < 86400:  # Meno di 1 giorno
            time_str = f"{int(time_diff.total_seconds() / 3600)} ore fa"
        else:
            time_str = f"{time_diff.days} giorni fa"
        
        pairs = ", ".join(session.get("trading_pairs", [])[:3])
        if len(session.get("trading_pairs", [])) > 3:
            pairs += "..."
            
        exchanges = ", ".join(session.get("selected_exchanges", [])[:2])
        
        return f"""
**Ultima sessione interrotta {time_str}**
- Exchange: {exchanges}
- Coppie: {pairs}
- Modalità AI: {'Attiva' if session.get('ai_mode') else 'Manuale'}
- Rischio: {session.get('risk_level', 'moderate').title()}
"""
    
    def clear_session(self) -> bool:
        """Pulisce file di sessione"""
        try:
            if os.path.exists(self.session_file):
                os.remove(self.session_file)
            return True
        except Exception as e:
            print(f"Errore pulizia sessione: {e}")
            return False
    
    def update_session_activity(self) -> bool:
        """Aggiorna timestamp ultima attività"""
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    session_data = json.load(f)
                
                session_data["last_active"] = datetime.now().isoformat()
                
                with open(self.session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Errore aggiornamento sessione: {e}")
            return False

class AutoResumeUI:
    """Componenti UI per auto-resume"""
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
    
    def show_resume_dialog(self, session: Dict[str, Any]) -> str:
        """Mostra dialog di ripresa nella UI Streamlit"""
        import streamlit as st
        
        st.info("🔄 Sessione Precedente Rilevata")
        
        # Mostra summary sessione precedente
        summary = self.session_manager.get_resume_summary(session)
        st.markdown(summary)
        
        # Opzioni per l'utente
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Riprendi Trading", use_container_width=True):
                return "RESUME"
        
        with col2:
            if st.button("🆕 Nuova Sessione", use_container_width=True):
                return "NEW"
        
        with col3:
            if st.button("⚙️ Modifica Config", use_container_width=True):
                return "MODIFY"
        
        return "WAITING"
    
    def show_resume_status(self, resumed: bool, session: Dict[str, Any] = None):
        """Mostra status della ripresa"""
        import streamlit as st
        
        if resumed and session:
            st.success(f"✅ Trading ripreso dalla sessione precedente")
            
            with st.expander("Dettagli Sessione Ripresa"):
                st.json({
                    "Ultima Attività": session.get("last_active"),
                    "Coppie Trading": session.get("trading_pairs"),
                    "Exchange": session.get("selected_exchanges"),
                    "Modalità AI": session.get("ai_mode"),
                    "ID Sessione": session.get("session_id")
                })
        
    def show_graceful_shutdown_ui(self):
        """UI per spegnimento sicuro"""
        import streamlit as st
        
        st.warning("⚠️ Spegnimento Sistema")
        st.write("Il sistema salverà lo stato e chiuderà le posizioni gradualmente.")
        
        if st.button("🛑 CONFERMA STOP SISTEMA", type="primary"):
            return True
        
        if st.button("❌ Annulla"):
            return False
        
        return None

# Integrazione con il sistema principale
def integrate_session_management():
    """Integra session management nel sistema esistente"""
    
    # Modifica simple_app.py per includere auto-resume
    session_integration_code = '''
# Aggiungi all'inizio di simple_app.py
from session_manager import SessionManager, AutoResumeUI

# Inizializzazione session manager
if 'session_manager' not in st.session_state:
    st.session_state.session_manager = SessionManager()
    st.session_state.auto_resume_ui = AutoResumeUI(st.session_state.session_manager)

# Controllo auto-resume all'inizio della app
def check_auto_resume():
    should_resume, session = st.session_state.session_manager.should_auto_resume()
    
    if should_resume and 'resume_handled' not in st.session_state:
        st.session_state.resume_available = True
        st.session_state.previous_session = session
        st.session_state.resume_handled = True
    
    return should_resume, session

# Salvataggio stato durante trading attivo
def save_current_state(config, portfolio_state=None):
    """Salva stato corrente per auto-resume"""
    trading_active = st.session_state.get('trading_active', False)
    st.session_state.session_manager.save_session_state(
        trading_active, config, portfolio_state
    )

# Gestione spegnimento sicuro
def handle_graceful_shutdown():
    """Gestisce spegnimento sicuro del sistema"""
    # Chiudi posizioni aperte gradualmente
    # Salva stato finale
    # Disattiva auto-resume per questa sessione
    st.session_state.session_manager.clear_session()
    st.success("Sistema spento in sicurezza")
'''
    
    return session_integration_code

if __name__ == "__main__":
    # Test del session manager
    sm = SessionManager()
    
    # Simula salvataggio sessione
    test_config = {
        "trading_pairs": ["BTC/USDT", "ETH/USDT"],
        "exchanges": ["Binance", "KuCoin"],
        "ai_auto_management": True,
        "risk_level": "moderate"
    }
    
    print("Testing session manager...")
    
    # Salva sessione di test
    saved = sm.save_session_state(True, test_config, {"btc": 1000})
    print(f"Sessione salvata: {saved}")
    
    # Carica sessione
    session = sm.load_previous_session()
    print(f"Sessione caricata: {session}")
    
    # Controlla auto-resume
    should_resume, session_data = sm.should_auto_resume()
    print(f"Dovrebbe riprendere: {should_resume}")
    
    if should_resume:
        summary = sm.get_resume_summary(session_data)
        print(f"Summary: {summary}")